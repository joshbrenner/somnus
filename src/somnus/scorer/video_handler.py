import os
import cv2
import numpy as np
import av

class VideoHandler:
    def __init__(self, video_path):
        self.video_path = video_path
        
        try:
            self.container = av.open(video_path)
            self.stream = self.container.streams.video[0]
            # Threading speeds up the PyAV decoding process
            self.stream.thread_type = 'AUTO' 
        except Exception as e:
            print(f"VideoHandler: Failed to open video with PyAV: {e}")
            self.container = None
            self.stream = None

        # Failsafe for missing/invalid video
        if not self.container:
            self.timestamps = None
            self.nominal_fps = 30.0
            self.last_frame = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(self.last_frame, "NO VIDEO", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return

        self.nominal_fps = float(self.stream.average_rate) if self.stream.average_rate else 30.0
        if self.nominal_fps < 1.0 or np.isnan(self.nominal_fps):
            self.nominal_fps = 30.0
            
        self.last_rendered_idx = -1
        self.current_time = -1.0
        self.last_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Initialization cascade: .npy -> PyAV extraction -> Fallback
        self.timestamps = self._load_or_extract_timestamps()
        
        # Setup the initial decoder generator
        self.decoder = self.container.decode(self.stream)

    def _load_or_extract_timestamps(self):
        """Attempts to load, then extract, then fallback for timestamps."""
        ts_path = os.path.splitext(self.video_path)[0] + '_timestamps.npy'
        if os.path.exists(ts_path):
            print(f"VideoHandler: Loaded exact hardware timestamps from {os.path.basename(ts_path)}")
            return np.load(ts_path)

        print("VideoHandler: No .npy found. Attempting PyAV exact timestamp extraction...")
        try:
            pts_list = []
            # Demuxing is incredibly fast because it skips video decoding
            for packet in self.container.demux(self.stream):
                if packet.pts is not None:
                    time_sec = float(packet.pts * self.stream.time_base)
                    pts_list.append(time_sec)
            
            # After fast-forwarding through packets, we MUST seek back to 0 for playback
            self.container.seek(0, stream=self.stream)
            
            if not pts_list:
                return self._generate_cfr_fallback()
                
            # Sort to guarantee presentation order (fixes issues if video contains B-frames)
            pts_list.sort()
            ts_array = np.array(pts_list, dtype=np.float64)
            
            # Auto-save so we don't have to do this again
            np.save(ts_path, ts_array)
            print(f"VideoHandler: Successfully extracted and saved {len(ts_array)} exact timestamps.")
            
            return ts_array
            
        except Exception as e:
            print(f"VideoHandler: PyAV extraction failed: {e}")
            return self._generate_cfr_fallback()

    def _generate_cfr_fallback(self):
        """Generates a linear array assuming perfectly stable framerate."""
        print("VideoHandler: Falling back to linear CFR timestamp generation.")
        duration = float(self.stream.duration * av.time_base) if self.stream.duration else 0.0
        frame_count = self.stream.frames or int(duration * self.nominal_fps)
        if frame_count <= 0: frame_count = 1000 
        return np.linspace(0.0, duration, frame_count, dtype=np.float64)

    def get_frame_at_time(self, exact_time_sec):
        """Uses PyAV to seek and decode the exact frame, ignoring OpenCV entirely."""
        if not self.container:
            return self.last_frame
            
        # 1. Map requested time to exact hardware timestamp
        if self.timestamps is not None and len(self.timestamps) > 0:
            idx = np.searchsorted(self.timestamps, exact_time_sec)
            if idx >= len(self.timestamps):
                idx = len(self.timestamps) - 1
            elif idx > 0:
                left_diff = exact_time_sec - self.timestamps[idx - 1]
                right_diff = self.timestamps[idx] - exact_time_sec
                idx = idx - 1 if left_diff < right_diff else idx
            target_time = self.timestamps[idx]
            target_idx = idx
        else:
            target_time = exact_time_sec
            target_idx = int(exact_time_sec * self.nominal_fps)

        # Avoid redundant decoding if the UI requests the same frame
        if target_idx == self.last_rendered_idx:
            return self.last_frame

        # 2. Seek if we are moving backward or jumping far ahead
        # Add a 0.1s tolerance to prevent constant backward seeking from VFR jitter
        if target_time < (self.current_time - 0.1) or (target_time - self.current_time) > 2.0:
            seek_time_sec = max(0.0, target_time - 0.5)
            seek_pts = int(seek_time_sec / self.stream.time_base)
            self.container.seek(seek_pts, stream=self.stream, backward=True)
            
            # Seeking invalidates the current decoder; we must recreate the generator
            self.decoder = self.container.decode(self.stream)
            self.current_time = -1.0 

        # 3. Read forward through frames until we hit the exact target timestamp
        try:
            for frame in self.decoder:
                self.current_time = float(frame.time)
                
                # Stop when we hit or slightly pass the target timestamp (using 5ms epsilon)
                if self.current_time >= target_time - 0.005: 
                    # Convert the raw PyAV frame to an OpenCV-compatible BGR array
                    self.last_frame = frame.to_ndarray(format='bgr24')
                    self.last_rendered_idx = target_idx
                    break
                    
        except av.EOFError:
            pass
        except Exception as e:
            print(f"VideoHandler decoding error: {e}")

        return self.last_frame

    def release(self):
        if self.container:
            self.container.close()
pub struct FrameTimeAnalyzer {
    frame: Vec<f32>,
    s_time: f32,
}

impl FrameTimeAnalyzer {
    pub fn new(length: usize) -> Self {
        FrameTimeAnalyzer {
            frame: vec![0.; length],
            s_time: 0.,
        }
    }

    pub fn add_frame_time(&mut self, time: f32) {
        self.frame.pop();
        self.frame.insert(0, time);
    }

    pub fn smooth_frame_time(&mut self) -> &f32 {
        self.s_time = self.frame.iter().sum::<f32>() / (self.frame.len() as f32);
        &self.s_time
    }
}

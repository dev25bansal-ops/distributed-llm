use std::collections::HashSet;
use std::process::Child;
use std::sync::Mutex;

pub struct AppState {
    pub coordinator: Mutex<Option<Child>>,
    pub worker: Mutex<Option<Child>>,
    pub cluster_id: Mutex<Option<String>>,
    pub api_port: Mutex<Option<u16>>,
    pub auth_token: Mutex<Option<String>>,
    pub active_downloads: Mutex<HashSet<String>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            coordinator: Mutex::new(None),
            worker: Mutex::new(None),
            cluster_id: Mutex::new(None),
            api_port: Mutex::new(None),
            auth_token: Mutex::new(None),
            active_downloads: Mutex::new(HashSet::new()),
        }
    }
}

// Kill child processes on drop to prevent zombies
impl Drop for AppState {
    fn drop(&mut self) {
        for child_opt in [&mut self.coordinator, &mut self.worker] {
            if let Ok(mut guard) = child_opt.lock() {
                if let Some(ref mut c) = *guard {
                    let _ = c.kill();
                    let _ = c.wait();
                }
            }
        }
    }
}

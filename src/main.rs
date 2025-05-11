#![allow(deprecated)]
use cube_app::run_app_core;
use env_logger::Builder;
use log::info;
use std::sync::Arc;

use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::window::WindowAttributes;

fn main() {
    Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting native application...");

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let window = Arc::new(
        event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Native Cube App")
                    .with_inner_size(LogicalSize::new(1024, 768)),
            )
            .expect("Failed to create window"),
    );

    pollster::block_on(run_app_core(window, event_loop));

    info!("Native application finished.");
}

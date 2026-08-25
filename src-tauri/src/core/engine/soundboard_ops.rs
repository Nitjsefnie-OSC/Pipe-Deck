use std::path::Path;

use crate::config::ConfigStore;
use crate::core::soundboard;

use super::{CoreEngine, EngineError};

impl CoreEngine {
    /// Plays `path` into `target_device_id`'s underlying device at full
    /// volume (a virtual input or a hardware input passthrough). Fire-and-
    /// forget — see `AudioBackend::play_sound` for what that does and
    /// doesn't guarantee.
    pub fn play_sound(&self, path: &Path, target_device_id: &str) -> Result<(), EngineError> {
        let device = self
            .graph
            .devices
            .iter()
            .find(|device| device.id == target_device_id)
            .ok_or_else(|| {
                EngineError::NotFound(format!("device not found: {target_device_id}"))
            })?;

        self.adapter
            .play_sound(path, &device.system_name, 100)
            .map_err(|error| EngineError::Adapter(error.to_string()))
    }

    /// Plays a Soundboard clip (#127) by re-resolving it fresh from config +
    /// disk rather than trusting a client-supplied path: loads the board,
    /// re-lists its folder (so only a file that's actually a direct, still-
    /// present child of the configured folder can ever be played — `clip_id`
    /// crosses the IPC boundary as a plain string), and plays it through the
    /// board's own destinations (#398's `target`/`monitor` — board-wide, not
    /// per-clip).
    ///
    /// A clip can play on either or both of two independent legs — `target`
    /// (what other people/apps hear, e.g. a virtual mic) and `monitor` (a
    /// local output so the user can hear/test the clip themselves) — each
    /// with its own volume. Both are attempted if configured; if both are
    /// configured and one fails, the other still plays and the failure is
    /// still surfaced (not silently swallowed). Errors if the board/clip
    /// doesn't exist, or if the board has neither leg configured yet.
    pub fn play_soundboard_clip(&self, board_id: &str, clip_id: &str) -> Result<(), EngineError> {
        let config = ConfigStore::new()
            .load_config()
            .map_err(|error| EngineError::Config(error.to_string()))?;
        let board = config
            .preferences
            .soundboard_boards
            .into_iter()
            .find(|board| board.id == board_id)
            .ok_or_else(|| {
                EngineError::NotFound(format!("soundboard board not found: {board_id}"))
            })?;

        let clips = soundboard::list_sounds(Path::new(&board.folder))
            .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
        let clip = clips
            .into_iter()
            .find(|clip| clip.id == clip_id)
            .ok_or_else(|| EngineError::NotFound(format!("clip not found: {clip_id}")))?;

        if board.target_system_name.is_none() && board.monitor_system_name.is_none() {
            return Err(EngineError::InvalidInput(format!(
                "\"{}\" tab has no target or monitor device set yet",
                board.name
            )));
        }

        let mut errors = Vec::new();
        if let Some(target) = &board.target_system_name {
            if let Err(error) =
                self.adapter
                    .play_sound(Path::new(&clip.path), target, board.target_volume_percent)
            {
                errors.push(format!("target: {error}"));
            }
        }
        if let Some(monitor) = &board.monitor_system_name {
            if let Err(error) = self.adapter.play_sound(
                Path::new(&clip.path),
                monitor,
                board.monitor_volume_percent,
            ) {
                errors.push(format!("monitor: {error}"));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(EngineError::Adapter(errors.join("; ")))
        }
    }

    /// Interrupts whatever Soundboard clip is currently playing (#399) —
    /// thin passthrough to `AudioBackend::stop_sound`, which owns the
    /// actual process handle(s) (see PD-036's rationale for that split).
    pub fn stop_soundboard_clip(&self) -> Result<(), EngineError> {
        self.adapter
            .stop_sound()
            .map_err(|error| EngineError::Adapter(error.to_string()))
    }
}

#[cfg(test)]
mod live_tests {
    //! `#[ignore]`d: hits a real PipeWire session, same rationale as
    //! `virtual_ops::live_tests`. Creates and tears down its own disposable
    //! virtual input.
    use super::*;

    #[test]
    #[ignore]
    fn play_sound_starts_playback_into_a_real_virtual_input() {
        assert_ne!(std::env::var("PIPE_DECK_USE_MOCK").as_deref(), Ok("1"));

        let mut engine = CoreEngine::new();
        engine.refresh_graph().expect("initial graph refresh");

        let created = engine
            .create_virtual_input("Pipe Deck Soundboard Playback Test")
            .expect("create disposable test device");

        let clip = Path::new("/usr/share/sounds/speech-dispatcher/test.wav");
        assert!(
            clip.is_file(),
            "expected a system test wav to exist at {}",
            clip.display()
        );

        let result = engine.play_sound(clip, &created.device_id);

        let _ = engine.remove_virtual_device(&created.system_name);

        result.expect("play_sound should succeed against a real virtual input");
    }

    #[test]
    #[ignore]
    fn play_soundboard_clip_plays_both_target_and_monitor_legs() {
        if let Err(error) = run_pipe_deck_407_probe() {
            panic!("{error}");
        }
    }

    use std::ffi::OsString;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    const PIPE_DECK_407_BASE: &str = "cedab6e2cf00175acf0a81cec76e42b878218351";
    const PROBE_RUN_DIR_ENV: &str = "PIPE_DECK_407_RUN_DIR";
    const PROBE_EXPECTED_HEAD_ENV: &str = "PIPE_DECK_407_EXPECTED_HEAD";
    const PROBE_CONFIG_DIR_ENV: &str = "PIPE_DECK_CONFIG_DIR";
    const PROBE_WRAPPER_LOG_ENV: &str = "PIPE_DECK_PW_CAT_LOG";
    const PROBE_TARGET_BOARD: &str = "pipe-deck-407-volume-board";
    const PROBE_CLIP: &str = "pipe-deck-407-fixture.wav";
    const STREAMING_WAIT: Duration = Duration::from_secs(5);
    const PLAYBACK_WAIT: Duration = Duration::from_secs(10);

    struct EnvRestore {
        values: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvRestore {
        fn capture(keys: &[&'static str]) -> Self {
            Self {
                values: keys
                    .iter()
                    .map(|key| (*key, std::env::var_os(key)))
                    .collect(),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (key, value) in &self.values {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    struct CaptureProcess {
        child: Child,
        log_path: PathBuf,
        node_name: String,
    }

    struct ProbeSetupCleanup {
        config_dir: PathBuf,
        wrapper_path: PathBuf,
        bin_dir: PathBuf,
        armed: bool,
    }

    impl ProbeSetupCleanup {
        fn new(config_dir: PathBuf, wrapper_path: PathBuf, bin_dir: PathBuf) -> Self {
            Self {
                config_dir,
                wrapper_path,
                bin_dir,
                armed: true,
            }
        }

        fn disarm(&mut self) {
            self.armed = false;
        }
    }

    impl Drop for ProbeSetupCleanup {
        fn drop(&mut self) {
            if !self.armed {
                return;
            }
            let _ = fs::remove_dir_all(&self.config_dir);
            let _ = fs::remove_file(&self.wrapper_path);
            let _ = fs::remove_dir(&self.bin_dir);
        }
    }

    fn probe_failure(message: impl std::fmt::Display) -> String {
        format!("infrastructure-inconclusive: {message}")
    }

    fn run_pipe_deck_407_probe() -> Result<(), String> {
        let run_dir = std::env::var_os(PROBE_RUN_DIR_ENV)
            .map(PathBuf::from)
            .ok_or_else(|| {
                probe_failure(format!(
                    "{PROBE_RUN_DIR_ENV} must be set by the dedicated runner"
                ))
            })?;
        fs::create_dir_all(&run_dir)
            .map_err(|error| probe_failure(format!("create run directory: {error}")))?;

        if Path::new("/dev/snd").exists() {
            return Err(probe_failure(
                "/dev/snd is present; hardware state is not allowed for this software-only probe",
            ));
        }
        if std::env::var_os("PIPE_DECK_USE_MOCK").is_some() {
            return Err(probe_failure("PIPE_DECK_USE_MOCK must be unset"));
        }
        if std::env::var_os("PIPEWIRE_REMOTE") != Some(OsString::from("pipewire-0")) {
            return Err(probe_failure("PIPEWIRE_REMOTE must be exactly pipewire-0"));
        }
        if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
            return Err(probe_failure(
                "XDG_RUNTIME_DIR must be set to the private runner runtime",
            ));
        }
        if std::env::var_os("PIPEWIRE_RUNTIME_DIR").is_none() {
            return Err(probe_failure(
                "PIPEWIRE_RUNTIME_DIR must be set to the private runner runtime",
            ));
        }
        if std::env::var_os("DISABLE_RTKIT") != Some(OsString::from("1")) {
            return Err(probe_failure("DISABLE_RTKIT must be exactly 1"));
        }

        let _config_env_guard = crate::config::store::lock_config_dir_env();
        let _env_restore = EnvRestore::capture(&[
            "PATH",
            PROBE_CONFIG_DIR_ENV,
            "PIPE_DECK_USE_MOCK",
            PROBE_WRAPPER_LOG_ENV,
            PROBE_EXPECTED_HEAD_ENV,
        ]);
        std::env::remove_var("PIPE_DECK_USE_MOCK");

        let helper_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../diagnostics/pipe-deck-407");
        let helper = helper_dir.join("probe.py");
        let wrapper_source = helper_dir.join("pw-cat-wrapper.sh");
        if !helper.is_file() || !wrapper_source.is_file() {
            return Err(probe_failure(format!(
                "missing diagnostic helper(s): {} and {}",
                helper.display(),
                wrapper_source.display()
            )));
        }

        let config_dir = run_dir.join("pipe-deck-config");
        let sounds_dir = run_dir.join("soundboard-fixture");
        let bin_dir = run_dir.join("bin");
        let wrapper_path = bin_dir.join("pw-cat");
        match fs::symlink_metadata(&config_dir) {
            Ok(_) => {
                return Err(probe_failure(format!(
                    "refusing to use pre-existing config directory {}",
                    config_dir.display()
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(probe_failure(format!(
                    "inspect config directory {}: {error}",
                    config_dir.display()
                )))
            }
        }
        fs::create_dir(&config_dir)
            .map_err(|error| probe_failure(format!("create config directory: {error}")))?;
        let mut setup_cleanup =
            ProbeSetupCleanup::new(config_dir.clone(), wrapper_path.clone(), bin_dir.clone());
        fs::create_dir_all(&sounds_dir)
            .map_err(|error| probe_failure(format!("create fixture directory: {error}")))?;
        std::env::set_var(PROBE_CONFIG_DIR_ENV, &config_dir);

        let wrapper_log = run_dir.join("pw-cat-wrapper.log");
        File::create(&wrapper_log)
            .map_err(|error| probe_failure(format!("create wrapper log: {error}")))?;
        std::env::set_var(PROBE_WRAPPER_LOG_ENV, &wrapper_log);
        fs::create_dir_all(&bin_dir)
            .map_err(|error| probe_failure(format!("create wrapper bin directory: {error}")))?;
        install_wrapper(&wrapper_source, &wrapper_path)?;
        prepend_path(&bin_dir)?;

        write_provenance(&run_dir, &helper)?;

        let fixture = sounds_dir.join(PROBE_CLIP);
        run_python(
            &helper,
            &[
                "fixture",
                "--output",
                fixture.to_string_lossy().as_ref(),
                "--metadata",
                run_dir
                    .join("fixture-metadata.json")
                    .to_string_lossy()
                    .as_ref(),
            ],
        )?;
        run_python(
            &helper,
            &[
                "marker",
                "--output",
                run_dir.join("target-marker.wav").to_string_lossy().as_ref(),
                "--metadata",
                run_dir
                    .join("target-marker-metadata.json")
                    .to_string_lossy()
                    .as_ref(),
                "--frequency",
                "733",
            ],
        )?;
        run_python(
            &helper,
            &[
                "marker",
                "--output",
                run_dir
                    .join("monitor-marker.wav")
                    .to_string_lossy()
                    .as_ref(),
                "--metadata",
                run_dir
                    .join("monitor-marker-metadata.json")
                    .to_string_lossy()
                    .as_ref(),
                "--frequency",
                "1237",
            ],
        )?;

        let mut engine = CoreEngine::new();
        engine
            .refresh_graph()
            .map_err(|error| probe_failure(format!("initial graph refresh: {error}")))?;

        let target = engine
            .create_virtual_output("Pipe Deck #407 volume probe target")
            .map_err(|error| probe_failure(format!("create native target output: {error}")))?;
        let monitor = match engine.create_virtual_output("Pipe Deck #407 volume probe monitor") {
            Ok(device) => device,
            Err(error) => {
                let _ = engine.remove_virtual_device(&target.system_name);
                return Err(probe_failure(format!(
                    "create native monitor output: {error}"
                )));
            }
        };
        let target_name = target.system_name.clone();
        let monitor_name = monitor.system_name.clone();
        if target_name == monitor_name {
            let _ = engine.remove_virtual_device(&target_name);
            return Err(probe_failure(
                "the two returned system_name values were identical",
            ));
        }
        if let Err(error) = append_provenance_names(&run_dir, &target_name, &monitor_name) {
            let _ = engine.remove_virtual_device(&target_name);
            let _ = engine.remove_virtual_device(&monitor_name);
            return Err(error);
        }

        let body_result = run_probe_body(
            &helper,
            &run_dir,
            &fixture,
            &mut engine,
            &target_name,
            &monitor_name,
            &wrapper_log,
        );
        let target_cleanup = engine
            .remove_virtual_device(&target_name)
            .map_err(|error| probe_failure(format!("remove target virtual output: {error}")));
        let monitor_cleanup = engine
            .remove_virtual_device(&monitor_name)
            .map_err(|error| probe_failure(format!("remove monitor virtual output: {error}")));
        let config_cleanup = fs::remove_dir_all(&config_dir)
            .map_err(|error| probe_failure(format!("remove owned config directory: {error}")));
        let wrapper_cleanup = cleanup_wrapper_path(&wrapper_path, &bin_dir);

        let cleanup_error = [
            target_cleanup,
            monitor_cleanup,
            config_cleanup,
            wrapper_cleanup,
        ]
        .into_iter()
        .find_map(Result::err);
        let cleanup_ok = cleanup_error.is_none();
        let result = match (body_result, cleanup_error) {
            (Err(error), Some(cleanup)) => Err(format!("{error}; cleanup also failed: {cleanup}")),
            (Err(error), None) => Err(error),
            (Ok(()), Some(cleanup)) => Err(cleanup),
            (Ok(()), None) => Ok(()),
        };
        if cleanup_ok {
            setup_cleanup.disarm();
        }
        result
    }

    fn install_wrapper(source: &Path, destination: &Path) -> Result<(), String> {
        if destination.exists() {
            return Err(probe_failure(format!(
                "refusing to overwrite existing wrapper path {}",
                destination.display()
            )));
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(source, destination).map_err(|error| {
                probe_failure(format!("install PATH-front pw-cat wrapper: {error}"))
            })?;
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = source;
            let _ = destination;
            Err(probe_failure(
                "the PipeWire diagnostic runner requires a Unix PATH wrapper",
            ))
        }
    }

    fn prepend_path(bin_dir: &Path) -> Result<(), String> {
        let mut paths = vec![bin_dir.to_path_buf()];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        let path = std::env::join_paths(paths).map_err(|error| {
            probe_failure(format!("construct PATH-front wrapper path: {error}"))
        })?;
        std::env::set_var("PATH", path);
        Ok(())
    }

    fn cleanup_wrapper_path(wrapper_path: &Path, bin_dir: &Path) -> Result<(), String> {
        if wrapper_path.exists() {
            fs::remove_file(wrapper_path)
                .map_err(|error| probe_failure(format!("remove owned wrapper symlink: {error}")))?;
        }
        if bin_dir.exists() {
            fs::remove_dir(bin_dir).map_err(|error| {
                probe_failure(format!("remove owned wrapper directory: {error}"))
            })?;
        }
        Ok(())
    }

    fn run_probe_body(
        helper: &Path,
        run_dir: &Path,
        fixture: &Path,
        engine: &mut CoreEngine,
        target_name: &str,
        monitor_name: &str,
        wrapper_log: &Path,
    ) -> Result<(), String> {
        verify_native_graph(run_dir, target_name, monitor_name)?;
        run_marker_phase(helper, run_dir, target_name, monitor_name, "target")?;
        run_marker_phase(helper, run_dir, target_name, monitor_name, "monitor")?;

        let mut board = soundboard::SoundboardBoard {
            id: PROBE_TARGET_BOARD.into(),
            name: "Pipe Deck #407 volume probe".into(),
            folder: fixture
                .parent()
                .ok_or_else(|| probe_failure("fixture has no parent directory"))?
                .display()
                .to_string(),
            target_system_name: Some(target_name.to_string()),
            target_volume_percent: 100,
            monitor_system_name: Some(monitor_name.to_string()),
            monitor_volume_percent: 100,
        };
        ConfigStore::new()
            .ensure_layout()
            .map_err(|error| probe_failure(format!("create isolated config layout: {error}")))?;

        for volume in [100_u8, 10_u8] {
            board.target_volume_percent = volume;
            board.monitor_volume_percent = volume;
            ConfigStore::new()
                .save_soundboard_board(board.clone())
                .map_err(|error| {
                    probe_failure(format!("save probe board at {volume}%: {error}"))
                })?;
            run_volume_phase(
                helper,
                run_dir,
                fixture,
                engine,
                target_name,
                monitor_name,
                wrapper_log,
                volume,
            )?;
        }

        run_python(
            helper,
            &[
                "wrapper",
                "--log",
                wrapper_log.to_string_lossy().as_ref(),
                "--target",
                target_name,
                "--monitor",
                monitor_name,
                "--json",
                run_dir
                    .join("wrapper-metrics.json")
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir
                    .join("wrapper-metrics.txt")
                    .to_string_lossy()
                    .as_ref(),
            ],
        )?;
        run_ratio_helper(
            helper,
            &[
                "ratio",
                "--target-100",
                run_dir
                    .join("target-100-metrics.json")
                    .to_string_lossy()
                    .as_ref(),
                "--target-10",
                run_dir
                    .join("target-10-metrics.json")
                    .to_string_lossy()
                    .as_ref(),
                "--monitor-100",
                run_dir
                    .join("monitor-100-metrics.json")
                    .to_string_lossy()
                    .as_ref(),
                "--monitor-10",
                run_dir
                    .join("monitor-10-metrics.json")
                    .to_string_lossy()
                    .as_ref(),
                "--json",
                run_dir
                    .join("volume-ratios.json")
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir.join("volume-ratios.txt").to_string_lossy().as_ref(),
            ],
        )
    }

    fn is_commit_sha(value: &str) -> bool {
        value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn write_provenance(run_dir: &Path, helper: &Path) -> Result<(), String> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let expected_head = std::env::var(PROBE_EXPECTED_HEAD_ENV).map_err(|_| {
            probe_failure(format!(
                "{PROBE_EXPECTED_HEAD_ENV} must be supplied by the runner"
            ))
        })?;
        if !is_commit_sha(&expected_head) {
            return Err(probe_failure(format!(
                "{PROBE_EXPECTED_HEAD_ENV} must be a 40-character hexadecimal commit SHA, got {expected_head:?}"
            )));
        }
        let revision = command_text(repo_root, "git", &["rev-parse", "HEAD"])?;
        let revision = revision.trim();
        if !is_commit_sha(revision) {
            return Err(probe_failure(format!(
                "git rev-parse HEAD did not return a 40-character hexadecimal commit SHA: {revision:?}"
            )));
        }
        if revision != expected_head.as_str() {
            return Err(probe_failure(format!(
                "tested HEAD {revision} does not match runner-supplied {PROBE_EXPECTED_HEAD_ENV} {expected_head}"
            )));
        }
        let ancestry = Command::new("git")
            .current_dir(repo_root)
            .args(["merge-base", "--is-ancestor", PIPE_DECK_407_BASE, revision])
            .status()
            .map_err(|error| probe_failure(format!("check probe base ancestry: {error}")))?;
        if !ancestry.success() {
            return Err(probe_failure(format!(
                "tested HEAD {revision} is not a descendant of probe base {PIPE_DECK_407_BASE}"
            )));
        }
        let pw_cli_version = command_text(Path::new("/"), "pw-cli", &["--version"])?;
        let pw_cat_version = command_text(Path::new("/"), "/usr/bin/pw-cat", &["--version"])?;
        let pw_record_version = command_text(Path::new("/"), "/usr/bin/pw-record", &["--version"])?;
        let provenance = format!(
            "git_head={revision}\nexpected_head={expected_head}\nbase_sha={PIPE_DECK_407_BASE}\nbase_is_ancestor=true\nhelper={}\npipewire_remote={}\nxdg_runtime_dir={}\ndisable_rtkit={}\n/dev/snd_present={}\npw_cli_version={}pw_cat_version={}pw_record_version={}",
            helper.display(),
            std::env::var("PIPEWIRE_REMOTE").unwrap_or_default(),
            std::env::var("XDG_RUNTIME_DIR").unwrap_or_default(),
            std::env::var("DISABLE_RTKIT").unwrap_or_default(),
            Path::new("/dev/snd").exists(),
            pw_cli_version,
            pw_cat_version,
            pw_record_version,
        );
        fs::write(run_dir.join("probe-provenance.txt"), provenance)
            .map_err(|error| probe_failure(format!("write human-readable provenance: {error}")))?;
        Ok(())
    }

    fn verify_native_graph(
        run_dir: &Path,
        target_name: &str,
        monitor_name: &str,
    ) -> Result<(), String> {
        let first_nodes = crate::backend::linux::pw_virtual_device_native::list_nodes()
            .ok_or_else(|| probe_failure("native PipeWire registry unavailable; creation may have fallen back through pactl"))?;
        if !native_nodes_contain(&first_nodes, target_name, monitor_name) {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut found = false;
            while Instant::now() < deadline {
                if let Some(nodes) = crate::backend::linux::pw_virtual_device_native::list_nodes() {
                    if native_nodes_contain(&nodes, target_name, monitor_name) {
                        found = true;
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(100));
            }
            if !found {
                return Err(probe_failure(
                    "native registry did not expose both returned system_name nodes",
                ));
            }
        }

        let dump_path = run_dir.join("pw-dump-sinks.json");
        let dump_stderr = run_dir.join("pw-dump-sinks.stderr");
        command_to_files("pw-dump", &[], &dump_path, &dump_stderr)?;
        let ports_path = run_dir.join("pw-link-iol.txt");
        let ports_stderr = run_dir.join("pw-link-iol.stderr");
        command_to_files("pw-link", &["-iol"], &ports_path, &ports_stderr)?;
        let helper =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../diagnostics/pipe-deck-407/probe.py");
        run_python(
            &helper,
            &[
                "verify-graph",
                "--dump",
                dump_path.to_string_lossy().as_ref(),
                "--ports",
                ports_path.to_string_lossy().as_ref(),
                "--target",
                target_name,
                "--monitor",
                monitor_name,
                "--json",
                run_dir
                    .join("graph-metrics.json")
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir.join("graph-metrics.txt").to_string_lossy().as_ref(),
            ],
        )
    }

    fn append_provenance_names(
        run_dir: &Path,
        target_name: &str,
        monitor_name: &str,
    ) -> Result<(), String> {
        let path = run_dir.join("probe-provenance.txt");
        let mut output = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|error| probe_failure(format!("open provenance for node names: {error}")))?;
        writeln!(output, "target_system_name={target_name}")
            .and_then(|_| writeln!(output, "monitor_system_name={monitor_name}"))
            .map_err(|error| probe_failure(format!("append node names to provenance: {error}")))
    }

    fn native_nodes_contain(
        nodes: &[crate::backend::linux::pw_virtual_device_native::NodeInfo],
        target_name: &str,
        monitor_name: &str,
    ) -> bool {
        let wanted = [target_name, monitor_name];
        wanted.iter().all(|name| {
            nodes.iter().any(|node| {
                node.system_name == *name && node.media_class.as_deref() == Some("Audio/Sink")
            })
        })
    }

    fn run_marker_phase(
        helper: &Path,
        run_dir: &Path,
        target_name: &str,
        monitor_name: &str,
        expected_destination: &str,
    ) -> Result<(), String> {
        let suffix = format!("marker-{expected_destination}");
        let target_capture_name = format!("pipe-deck-407-capture-target-{suffix}");
        let monitor_capture_name = format!("pipe-deck-407-capture-monitor-{suffix}");
        let mut captures = start_capture_pair(
            run_dir,
            target_name,
            monitor_name,
            &target_capture_name,
            &monitor_capture_name,
            &suffix,
        )?;
        let phase_result = (|| {
            verify_capture_links(
                helper,
                run_dir,
                target_name,
                monitor_name,
                &target_capture_name,
                &monitor_capture_name,
                &suffix,
            )?;
            let marker_path = run_dir.join(format!("{expected_destination}-marker.wav"));
            play_marker(
                &marker_path,
                if expected_destination == "target" {
                    target_name
                } else {
                    monitor_name
                },
                run_dir,
                &suffix,
            )?;
            finish_capture_pair(&mut captures)
        })();
        if phase_result.is_err() {
            cleanup_capture_pair(&mut captures);
        }
        phase_result?;

        let expected_raw = run_dir.join(if expected_destination == "target" {
            format!("target-{suffix}.raw")
        } else {
            format!("monitor-{suffix}.raw")
        });
        let other_destination = if expected_destination == "target" {
            "monitor"
        } else {
            "target"
        };
        let other_raw = run_dir.join(if other_destination == "target" {
            format!("target-{suffix}.raw")
        } else {
            format!("monitor-{suffix}.raw")
        });
        run_python(
            helper,
            &[
                "marker-check",
                "--expected",
                expected_raw.to_string_lossy().as_ref(),
                "--other",
                other_raw.to_string_lossy().as_ref(),
                "--destination",
                expected_destination,
                "--json",
                run_dir
                    .join(format!("marker-{expected_destination}-metrics.json"))
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir
                    .join(format!("marker-{expected_destination}-metrics.txt"))
                    .to_string_lossy()
                    .as_ref(),
            ],
        )
    }

    fn play_marker(
        marker_path: &Path,
        target_name: &str,
        run_dir: &Path,
        suffix: &str,
    ) -> Result<(), String> {
        let log_path = run_dir.join(format!("marker-playback-{suffix}.log"));
        let output = Command::new("/usr/bin/pw-cat")
            .args(["--playback", "--target", target_name, "--volume", "1.00"])
            .arg(marker_path)
            .output()
            .map_err(|error| {
                probe_failure(format!("run target-only/monitor-only marker: {error}"))
            })?;
        let mut log = output.stdout;
        log.extend_from_slice(&output.stderr);
        fs::write(&log_path, log)
            .map_err(|error| probe_failure(format!("write marker playback log: {error}")))?;
        if !output.status.success() {
            return Err(probe_failure(format!(
                "marker playback exited with {} (see {})",
                format_status(output.status),
                log_path.display()
            )));
        }
        Ok(())
    }

    fn run_volume_phase(
        helper: &Path,
        run_dir: &Path,
        fixture: &Path,
        engine: &mut CoreEngine,
        target_name: &str,
        monitor_name: &str,
        wrapper_log: &Path,
        volume: u8,
    ) -> Result<(), String> {
        let suffix = volume.to_string();
        let target_capture_name = format!("pipe-deck-407-capture-target-{suffix}");
        let monitor_capture_name = format!("pipe-deck-407-capture-monitor-{suffix}");
        let mut captures = start_capture_pair(
            run_dir,
            target_name,
            monitor_name,
            &target_capture_name,
            &monitor_capture_name,
            &suffix,
        )?;
        let play_result = (|| {
            verify_capture_links(
                helper,
                run_dir,
                target_name,
                monitor_name,
                &target_capture_name,
                &monitor_capture_name,
                &suffix,
            )?;
            engine
                .play_soundboard_clip(PROBE_TARGET_BOARD, PROBE_CLIP)
                .map_err(|error| {
                    probe_failure(format!("play_soundboard_clip at {volume}%: {error}"))
                })?;
            wait_for_two_playback_exits(wrapper_log, volume)?;
            Ok(())
        })();
        let capture_result = finish_capture_pair(&mut captures);
        if play_result.is_err() || capture_result.is_err() {
            cleanup_capture_pair(&mut captures);
        }
        play_result?;
        capture_result?;

        let target_raw = run_dir.join(format!("target-{suffix}.raw"));
        let monitor_raw = run_dir.join(format!("monitor-{suffix}.raw"));
        run_python(
            helper,
            &[
                "metrics",
                "--raw",
                target_raw.to_string_lossy().as_ref(),
                "--volume",
                suffix.as_str(),
                "--json",
                run_dir
                    .join(format!("target-{suffix}-metrics.json"))
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir
                    .join(format!("target-{suffix}-metrics.txt"))
                    .to_string_lossy()
                    .as_ref(),
            ],
        )?;
        run_python(
            helper,
            &[
                "metrics",
                "--raw",
                monitor_raw.to_string_lossy().as_ref(),
                "--volume",
                suffix.as_str(),
                "--json",
                run_dir
                    .join(format!("monitor-{suffix}-metrics.json"))
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir
                    .join(format!("monitor-{suffix}-metrics.txt"))
                    .to_string_lossy()
                    .as_ref(),
            ],
        )
    }

    fn start_capture_pair(
        run_dir: &Path,
        target_sink: &str,
        monitor_sink: &str,
        target_capture_name: &str,
        monitor_capture_name: &str,
        suffix: &str,
    ) -> Result<Vec<CaptureProcess>, String> {
        let mut captures = Vec::new();
        let target = spawn_capture(
            run_dir,
            target_sink,
            target_capture_name,
            &format!("target-{suffix}"),
        )?;
        captures.push(target);
        let monitor = match spawn_capture(
            run_dir,
            monitor_sink,
            monitor_capture_name,
            &format!("monitor-{suffix}"),
        ) {
            Ok(capture) => capture,
            Err(error) => {
                cleanup_capture_pair(&mut captures);
                return Err(error);
            }
        };
        captures.push(monitor);
        if let Err(error) = wait_for_streaming(&captures) {
            cleanup_capture_pair(&mut captures);
            return Err(error);
        }
        Ok(captures)
    }

    fn spawn_capture(
        run_dir: &Path,
        sink_name: &str,
        capture_name: &str,
        stem: &str,
    ) -> Result<CaptureProcess, String> {
        let raw_path = run_dir.join(format!("{stem}.raw"));
        let log_path = run_dir.join(format!("record-{stem}.log"));
        let log = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)
            .map_err(|error| {
                probe_failure(format!("open recorder log {}: {error}", log_path.display()))
            })?;
        let stdout = log
            .try_clone()
            .map_err(|error| probe_failure(format!("clone recorder log handle: {error}")))?;
        let properties = format!(r#"{{"stream.capture.sink":true,"node.name":"{capture_name}"}}"#);
        let child = Command::new("timeout")
            .args([
                "--foreground",
                "--signal=TERM",
                "--kill-after=2s",
                "6s",
                "pw-record",
                "--verbose",
                "--raw",
                "--rate",
                "48000",
                "--channels",
                "2",
                "--format",
                "f32",
                "--target",
                sink_name,
                "--properties",
                &properties,
            ])
            .arg(&raw_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|error| probe_failure(format!("spawn pw-record for {sink_name}: {error}")))?;
        Ok(CaptureProcess {
            child,
            log_path,
            node_name: capture_name.to_string(),
        })
    }

    fn wait_for_streaming(captures: &[CaptureProcess]) -> Result<(), String> {
        for capture in captures {
            wait_for_log_text(&capture.log_path, "streaming", STREAMING_WAIT).map_err(|error| {
                probe_failure(format!(
                    "recorder {} did not reach STREAMING: {error}",
                    capture.node_name
                ))
            })?;
        }
        Ok(())
    }

    fn finish_capture_pair(captures: &mut [CaptureProcess]) -> Result<(), String> {
        let mut first_error = None;
        for capture in captures {
            match capture.child.wait() {
                Ok(status) if status.success() || status.code() == Some(124) => {}
                Ok(status) => {
                    first_error.get_or_insert_with(|| {
                        probe_failure(format!(
                            "recorder {} exited with {}",
                            capture.node_name,
                            format_status(status)
                        ))
                    });
                }
                Err(error) => {
                    first_error.get_or_insert_with(|| {
                        probe_failure(format!("wait for recorder {}: {error}", capture.node_name))
                    });
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn cleanup_capture_pair(captures: &mut [CaptureProcess]) {
        for capture in captures {
            match capture.child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => {
                    let _ = capture.child.kill();
                    let _ = capture.child.wait();
                }
            }
        }
    }

    fn verify_capture_links(
        helper: &Path,
        run_dir: &Path,
        target_sink: &str,
        monitor_sink: &str,
        target_capture: &str,
        monitor_capture: &str,
        suffix: &str,
    ) -> Result<(), String> {
        let links_path = run_dir.join(format!("pw-link-{suffix}-lI.txt"));
        let stderr_path = run_dir.join(format!("pw-link-{suffix}-lI.stderr"));
        command_to_files("pw-link", &["-lI"], &links_path, &stderr_path)?;
        run_python(
            helper,
            &[
                "verify-links",
                "--links",
                links_path.to_string_lossy().as_ref(),
                "--target-sink",
                target_sink,
                "--monitor-sink",
                monitor_sink,
                "--target-capture",
                target_capture,
                "--monitor-capture",
                monitor_capture,
                "--json",
                run_dir
                    .join(format!("links-{suffix}.json"))
                    .to_string_lossy()
                    .as_ref(),
                "--human",
                run_dir
                    .join(format!("links-{suffix}.txt"))
                    .to_string_lossy()
                    .as_ref(),
            ],
        )
    }

    fn wait_for_two_playback_exits(wrapper_log: &Path, volume: u8) -> Result<(), String> {
        let initial = read_text(wrapper_log)?;
        let initial_successes = initial.lines().filter(|line| *line == "exit=0").count();
        let expected_successes = initial_successes + 2;
        let deadline = Instant::now() + PLAYBACK_WAIT;
        loop {
            let text = read_text(wrapper_log)?;
            if text
                .lines()
                .any(|line| line.starts_with("exit=") && line != "exit=0")
            {
                return Err(probe_failure(format!(
                    "pw-cat child failed while playing at {volume}% (see {})",
                    wrapper_log.display()
                )));
            }
            let successes = text.lines().filter(|line| *line == "exit=0").count();
            if successes >= expected_successes {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(probe_failure(format!(
                    "timed out waiting for both pw-cat playback children at {volume}%"
                )));
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn wait_for_log_text(path: &Path, needle: &str, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            let text = read_text(path)?;
            if text
                .to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out waiting for {needle} in {}",
                    path.display()
                ));
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn read_text(path: &Path) -> Result<String, String> {
        fs::read_to_string(path)
            .map_err(|error| probe_failure(format!("read {}: {error}", path.display())))
    }

    fn format_status(status: ExitStatus) -> String {
        status
            .code()
            .map_or_else(|| "signal".to_string(), |code| code.to_string())
    }

    fn run_python(helper: &Path, args: &[&str]) -> Result<(), String> {
        let output = Command::new("python3")
            .arg(helper)
            .args(args)
            .output()
            .map_err(|error| probe_failure(format!("run {}: {error}", helper.display())))?;
        if output.status.success() {
            return Ok(());
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(probe_failure(format!(
            "{} {} failed (stdout: {}; stderr: {})",
            helper.display(),
            args.join(" "),
            stdout.trim(),
            stderr.trim()
        )))
    }

    fn run_ratio_helper(helper: &Path, args: &[&str]) -> Result<(), String> {
        let output = Command::new("python3")
            .arg(helper)
            .args(args)
            .output()
            .map_err(|error| probe_failure(format!("run volume ratio helper: {error}")))?;
        if output.status.success() {
            return Ok(());
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "volume-probe classification failure (see volume-ratios.json): stdout: {}; stderr: {}",
            stdout.trim(),
            stderr.trim()
        ))
    }

    fn command_text(current_dir: &Path, program: &str, args: &[&str]) -> Result<String, String> {
        let output = Command::new(program)
            .current_dir(current_dir)
            .args(args)
            .output()
            .map_err(|error| probe_failure(format!("run {program}: {error}")))?;
        if !output.status.success() {
            return Err(probe_failure(format!(
                "{program} {} exited with {}: {}",
                args.join(" "),
                format_status(output.status),
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn command_to_files(
        program: &str,
        args: &[&str],
        stdout_path: &Path,
        stderr_path: &Path,
    ) -> Result<(), String> {
        let output = Command::new(program)
            .args(args)
            .output()
            .map_err(|error| probe_failure(format!("run {program}: {error}")))?;
        fs::write(stdout_path, &output.stdout)
            .map_err(|error| probe_failure(format!("write {}: {error}", stdout_path.display())))?;
        fs::write(stderr_path, &output.stderr)
            .map_err(|error| probe_failure(format!("write {}: {error}", stderr_path.display())))?;
        if !output.status.success() {
            return Err(probe_failure(format!(
                "{program} {} exited with {} (see {})",
                args.join(" "),
                format_status(output.status),
                stderr_path.display()
            )));
        }
        Ok(())
    }

    #[test]
    fn play_soundboard_clip_errors_when_board_has_no_destination_configured() {
        let _guard = crate::config::store::lock_config_dir_env();
        let config_dir = std::env::temp_dir().join(format!(
            "pipe-deck-soundboard-ops-unit-test-config-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&config_dir);
        std::env::set_var("PIPE_DECK_CONFIG_DIR", &config_dir);
        std::env::set_var("PIPE_DECK_USE_MOCK", "1");

        let sounds_dir = std::env::temp_dir().join(format!(
            "pipe-deck-soundboard-ops-unit-test-sounds-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&sounds_dir);
        std::fs::create_dir_all(&sounds_dir).unwrap();
        std::fs::write(sounds_dir.join("untargeted.wav"), b"fake").unwrap();

        let mut engine = CoreEngine::new();
        engine
            .refresh_graph()
            .expect("mock refresh_graph should not fail");

        let board = soundboard::SoundboardBoard {
            id: "unit-test-board".into(),
            name: "Unit Test".into(),
            folder: sounds_dir.display().to_string(),
            target_system_name: None,
            target_volume_percent: 100,
            monitor_system_name: None,
            monitor_volume_percent: 100,
        };
        ConfigStore::new().ensure_layout().unwrap();
        ConfigStore::new().save_soundboard_board(board).unwrap();

        let result = engine.play_soundboard_clip("unit-test-board", "untargeted.wav");

        let _ = std::fs::remove_dir_all(&config_dir);
        let _ = std::fs::remove_dir_all(&sounds_dir);
        std::env::remove_var("PIPE_DECK_CONFIG_DIR");
        std::env::remove_var("PIPE_DECK_USE_MOCK");

        assert!(matches!(result, Err(EngineError::InvalidInput(_))));
    }

    #[test]
    fn play_soundboard_clip_plays_monitor_only_board_via_mock() {
        let _guard = crate::config::store::lock_config_dir_env();
        let config_dir = std::env::temp_dir().join(format!(
            "pipe-deck-soundboard-ops-unit-test-monitor-only-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&config_dir);
        std::env::set_var("PIPE_DECK_CONFIG_DIR", &config_dir);
        std::env::set_var("PIPE_DECK_USE_MOCK", "1");

        let sounds_dir = std::env::temp_dir().join(format!(
            "pipe-deck-soundboard-ops-unit-test-monitor-only-sounds-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&sounds_dir);
        std::fs::create_dir_all(&sounds_dir).unwrap();
        std::fs::write(sounds_dir.join("test-only.wav"), b"fake").unwrap();

        let mut engine = CoreEngine::new();
        engine
            .refresh_graph()
            .expect("mock refresh_graph should not fail");

        let board = soundboard::SoundboardBoard {
            id: "monitor-only-board".into(),
            name: "Monitor Only".into(),
            folder: sounds_dir.display().to_string(),
            target_system_name: None,
            target_volume_percent: 100,
            monitor_system_name: Some("pipe-deck-mock-monitor".to_string()),
            monitor_volume_percent: 60,
        };
        ConfigStore::new().ensure_layout().unwrap();
        ConfigStore::new().save_soundboard_board(board).unwrap();

        let result = engine.play_soundboard_clip("monitor-only-board", "test-only.wav");

        let _ = std::fs::remove_dir_all(&config_dir);
        let _ = std::fs::remove_dir_all(&sounds_dir);
        std::env::remove_var("PIPE_DECK_CONFIG_DIR");
        std::env::remove_var("PIPE_DECK_USE_MOCK");

        result.expect("a monitor-only board should still play");
    }

    #[test]
    fn play_soundboard_clip_errors_for_an_unknown_board() {
        let _guard = crate::config::store::lock_config_dir_env();
        let config_dir = std::env::temp_dir().join(format!(
            "pipe-deck-soundboard-ops-unit-test-unknown-board-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&config_dir);
        std::env::set_var("PIPE_DECK_CONFIG_DIR", &config_dir);
        std::env::set_var("PIPE_DECK_USE_MOCK", "1");

        let mut engine = CoreEngine::new();
        engine
            .refresh_graph()
            .expect("mock refresh_graph should not fail");

        let result = engine.play_soundboard_clip("no-such-board", "whatever.wav");

        let _ = std::fs::remove_dir_all(&config_dir);
        std::env::remove_var("PIPE_DECK_CONFIG_DIR");
        std::env::remove_var("PIPE_DECK_USE_MOCK");

        assert!(matches!(result, Err(EngineError::NotFound(_))));
    }

    #[test]
    fn stop_soundboard_clip_delegates_to_the_adapter() {
        let _guard = crate::config::store::lock_config_dir_env();
        std::env::set_var("PIPE_DECK_USE_MOCK", "1");

        let mut engine = CoreEngine::new();
        engine
            .refresh_graph()
            .expect("mock refresh_graph should not fail");

        let result = engine.stop_soundboard_clip();

        std::env::remove_var("PIPE_DECK_USE_MOCK");

        result.expect("stop_soundboard_clip should succeed against the mock backend");
    }
}

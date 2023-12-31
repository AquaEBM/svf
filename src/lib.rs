#![feature(portable_simd)]

extern crate alloc;

use editor::SVFBode;
use nih_plug_vizia::{create_vizia_editor, ViziaState, ViziaTheming};
use plugin_util::{
    filter::svf::{FilterMode, SVF},
    simd::*,
};

use nih_plug::prelude::*;
mod editor;

use alloc::sync::Arc;
use core::{f32::consts::TAU, sync::atomic::Ordering};

const MIN_FREQ: f32 = 13.;
const MAX_FREQ: f32 = 21000.;
const BASE_SAMPLE_RATE: f32 = 44100.;

const NUM_CHANNELS: usize = 2; // stereo

type Filter = SVF<NUM_CHANNELS>;

#[derive(Params)]
struct SVFParams {
    two_pi_tick: AtomicF32,
    #[persist = "editor_state"]
    vizia_state: Arc<ViziaState>,
    #[id = "cutoff"]
    cutoff: FloatParam,
    #[id = "res"]
    res: FloatParam,
    #[id = "gain"]
    gain: FloatParam,
    #[id = "mode"]
    mode: EnumParam<FilterMode>,
}

impl Default for SVFParams {
    fn default() -> Self {
        Self {
            two_pi_tick: AtomicF32::new(TAU / BASE_SAMPLE_RATE),
            vizia_state: ViziaState::new(|| (400, 140)),
            cutoff: FloatParam::new("Cutoff", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_value_to_string(Arc::new(|value| {
                    (MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(value)).to_string()
                })),

            res: FloatParam::new(
                "Resonance",
                1.,
                FloatRange::Reversed(&FloatRange::Skewed {
                    min: 0.02,
                    max: 1.,
                    factor: 0.37,
                }),
            ),

            gain: FloatParam::new(
                "Gain",
                0.,
                FloatRange::Linear {
                    min: -30.,
                    max: 30.,
                },
            )
            .with_unit(" db"),

            mode: EnumParam::new("Filter Mode", FilterMode::default()),
        }
    }
}

impl SVFParams {
    fn get_values(&self, two_pi_tick: f32) -> (f32x2, f32x2, f32x2, FilterMode) {
        let cutoff_normalized = self.cutoff.unmodulated_plain_value();
        let gain_normalized = self.gain.unmodulated_plain_value();
        (
            Simd::splat(two_pi_tick * MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(cutoff_normalized)),
            Simd::splat(2. * self.res.unmodulated_plain_value()),
            Simd::splat(10f32.powf(gain_normalized * (1. / 20.))),
            self.mode.unmodulated_plain_value(),
        )
    }
}

#[derive(Default)]
pub struct SVFFilter {
    params: Arc<SVFParams>,
    two_pi_tick: f32,
    min_smoothing_time: usize,
    filter: Filter,
}

impl Plugin for SVFFilter {
    const NAME: &'static str = "Linear SVF";

    const VENDOR: &'static str = "AquaEBM";

    const URL: &'static str = "monkey.com";

    const EMAIL: &'static str = "monke@monkey.com";

    const VERSION: &'static str = "0.6.9";

    const MIDI_INPUT: MidiConfig = MidiConfig::None;

    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    const HARD_REALTIME_ONLY: bool = false;

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(NUM_CHANNELS as u32),
        main_output_channels: NonZeroU32::new(NUM_CHANNELS as u32),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = ();

    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let (w_c, res, gain, mode) = self.params.get_values(self.two_pi_tick);
        let update = Filter::get_smoothing_update_function(mode);
        let get_output = Filter::get_output_function(mode);

        let f = &mut self.filter;

        let num_samples = buffer.samples().max(self.min_smoothing_time);
        update(f, w_c, res, gain, num_samples);

        for mut frame in buffer.iter_samples() {
            // SAFETY: we only support a stereo configuration so these indices are valid

            let sample = Simd::from_array(unsafe {
                [*frame.get_unchecked_mut(0), *frame.get_unchecked_mut(1)]
            });

            f.update_all_smoothers();
            f.process(sample);

            let sample = get_output(f);

            unsafe {
                *frame.get_unchecked_mut(0) = sample[0];
                *frame.get_unchecked_mut(1) = sample[1];
            }
        }

        ProcessStatus::Normal
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        create_vizia_editor(
            self.params.vizia_state.clone(),
            ViziaTheming::Builtin,
            move |cx, _gui_ctx| {
                SVFBode::new(cx, params.clone());
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {

        let sr = buffer_config.sample_rate;
        self.two_pi_tick = TAU / sr;

        self.min_smoothing_time = usize::max((sr / 1000.) as usize, 16);

        let (w_c, res, gain, mode) = self.params.get_values(self.two_pi_tick);
        let update = Filter::get_update_function(mode);

        update(&mut self.filter, w_c, res, gain);

        self.params
            .two_pi_tick
            .store(self.two_pi_tick, Ordering::Relaxed);
        true
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

impl Vst3Plugin for SVFFilter {
    const VST3_CLASS_ID: [u8; 16] = *b"svf_monkeeeeeeee";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Filter, Vst3SubCategory::Fx];
}

impl ClapPlugin for SVFFilter {
    const CLAP_ID: &'static str = "com.AquaEBM.linear_svf";

    const CLAP_DESCRIPTION: Option<&'static str> = Some("Linear SVF Filter");

    const CLAP_MANUAL_URL: Option<&'static str> = None;

    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Filter];
}

nih_export_clap!(SVFFilter);
nih_export_vst3!(SVFFilter);

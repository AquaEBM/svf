#![feature(portable_simd)]

use plugin_util::{
    util::map,
    smoothing::{LogSmoother, SIMDSmoother},
    filter::Integrator,
};

use nih_plug::prelude::*;
use core_simd::simd::*;

use std::{sync::Arc, array, f32::consts::PI};

#[derive(Params)]
pub struct SVFParams {
    #[id = "cutoff"]
    cutoff: FloatParam,
    #[id = "res"]
    res: FloatParam,
}

impl Default for SVFParams {
    fn default() -> Self {
        Self {

            cutoff: FloatParam::new(
                "Cutoff",
                660.,
                FloatRange::Skewed {
                    min: 13.,
                    max: 21000.,
                    factor: 0.2,
                },
            ),

            res: FloatParam::new(
                "Resonance",
                1.,
                FloatRange::Reversed(&FloatRange::Skewed {
                    min: 0.03,
                    max: 2.,
                    factor: 0.3,
                }),
            ),
        }
    }
}

#[derive(Default)]
pub struct SVF<const N: usize>
where
    LaneCount<N>: SupportedLaneCount
{
    params: Arc<SVFParams>,
    g: LogSmoother<N>,
    r: LogSmoother<N>,
    integrators: [Integrator<N> ; 2],
    pi_tick: Simd<f32, N>,
}

impl<const N: usize> SVF<N>
where
    LaneCount<N>: SupportedLaneCount
{
    fn update_smoothers(&mut self, block_len: usize) {
        let cutoff = Simd::splat(self.params.cutoff.unmodulated_plain_value());

        self.g.set_target(map(cutoff * self.pi_tick, f32::tan), block_len);

        let r = Simd::splat(self.params.res.unmodulated_plain_value());

        self.r.set_target(r, block_len);
    }

    fn process(&mut self, sample: Simd<f32, N>) -> Simd<f32, N> {

        self.g.tick();
        self.r.tick();

        let &g = self.g.current();
        let &r = self.r.current();

        let s1 = self.integrators[0].previous_output();
        let s2 = self.integrators[1].previous_output();

        let g1 = r + g;

        let hp = (sample - s2 - g1 * s1) / (Simd::splat(1.) + g * g1);

        let bp = self.integrators[0].process(hp, g);
        self.integrators[1].process(bp, g)
    }
}

impl<const N: usize> Plugin for SVF<N>
where
    LaneCount<N>: SupportedLaneCount
{
    const NAME: &'static str = "Linear SVF";

    const VENDOR: &'static str = "AquaEBM";

    const URL: &'static str = "monkey.com";

    const EMAIL: &'static str = "monke@monkey.com";

    const VERSION: &'static str = "0.6.9";

    const MIDI_INPUT: MidiConfig = MidiConfig::None;

    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    const HARD_REALTIME_ONLY: bool = false;

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(N as u32),
            main_output_channels: NonZeroU32::new(N as u32),
            ..AudioIOLayout::const_default()
        }
    ];

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

        self.update_smoothers(buffer.samples());

        for mut frame in buffer.iter_samples() {

            let mut sample = array::from_fn(
                |i| *unsafe { frame.get_unchecked_mut(i) }
            ).into();

            sample = self.process(sample);

            unsafe {
                *frame.get_unchecked_mut(0) = sample[0];
                *frame.get_unchecked_mut(1) = sample[1];
            }
        }

        ProcessStatus::Normal
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        None
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {

        self.pi_tick = Simd::splat(PI / buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        self.integrators[0].reset();
        self.integrators[1].reset();
    }
}

impl<const N: usize> Vst3Plugin for SVF<N>
where
    LaneCount<N>: SupportedLaneCount
{
    const VST3_CLASS_ID: [u8; 16] = *b"svf_monkeeeeeeee";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Filter,
        Vst3SubCategory::Fx,
    ];
}

impl<const N: usize> ClapPlugin for SVF<N>
where
    LaneCount<N>: SupportedLaneCount
{
    const CLAP_ID: &'static str = "com.AquaEBM.linear_svf";

    const CLAP_DESCRIPTION: Option<&'static str> = Some("Linear SVF Filter");

    const CLAP_MANUAL_URL: Option<&'static str> = None;

    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
    ];
}

nih_export_clap!(SVF<2>);
nih_export_vst3!(SVF<2>);
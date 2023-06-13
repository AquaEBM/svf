#![feature(portable_simd)]

use plugin_util::{filter::svf::SVF, simd::*};

use nih_plug::prelude::*;

use std::{sync::Arc, array};

const MIN_FREQ: f32 = 13.;
const MAX_FREQ: f32 = 21000.;

#[derive(Params)]
struct SVFParams {
    #[id = "cutoff"]
    cutoff: FloatParam,
    #[id = "res"]
    res: FloatParam,
    #[id = "gain"]
    gain: FloatParam,
    #[id = "mode"]
    mode: EnumParam<FilterMode>
}

#[derive(Enum, PartialEq, Eq)]
pub enum FilterMode {
    #[name = "Highpass"]
    HP,
    #[name = "Lowpass"]
    LP,
    #[name = "Bandpass"]
    BP,
    #[name = "Unit Bandpass"]
    BP1,
    #[name = "Allpass"]
    AP,
    #[name = "Notch"]
    NCH,
    #[name = "High Shelf"]
    HSH,
    #[name = "Band shelf"]
    BSH,
    #[name = "Low shelf"]
    LSH,
    #[name = "peaking"]
    PK,
}

impl FilterMode {
    pub fn output_function<const N: usize>(&self) -> fn(&SVF<N>) -> Simd<f32, N>
    where
        LaneCount<N>: SupportedLaneCount
    {
        match self {
            FilterMode::AP => SVF::<N>::get_allpass,
            FilterMode::HP => SVF::<N>::get_highpass,
            FilterMode::LP => SVF::<N>::get_lowpass,
            FilterMode::BP => SVF::<N>::get_bandpass,
            FilterMode::BP1 => SVF::<N>::get_bandpass1,
            FilterMode::NCH => SVF::<N>::get_notch,
            FilterMode::HSH => SVF::<N>::get_high_shelf,
            FilterMode::BSH => SVF::<N>::get_band_shelf,
            FilterMode::LSH => SVF::<N>::get_low_shelf,
            FilterMode::PK => SVF::<N>::get_peaking,
        }
    }
}

impl Default for SVFParams {
    fn default() -> Self {
        Self {

            cutoff: FloatParam::new(
                "Cutoff",
                0.5,
                FloatRange::Linear {
                    min: 0.,
                    max: 1.,
               },
            ).with_value_to_string(Arc::new(
                |value| (MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(value)).to_string()
            )),

            res: FloatParam::new(
                "Resonance",
                1.,
                FloatRange::Reversed(&FloatRange::Skewed {
                    min: 0.03,
                    max: 2.,
                    factor: 0.3,
                }),
            ),

            gain: FloatParam::new(
                "Gain",
                1.,
                FloatRange::Linear { min: -18., max: 18. }
            ),

            mode: EnumParam::new("Filter Mode", FilterMode::AP),
        }
    }
}

#[derive(Default)]
pub struct SVFFilter {
    params: Arc<SVFParams>,
    filter: SVF<2>,
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

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
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

        let block_len = buffer.samples();

        let cutoff = self.params.cutoff.unmodulated_plain_value();

        self.filter.set_params_smoothed(
            Simd::splat(MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(cutoff)),
            Simd::splat(self.params.res.unmodulated_plain_value()),
            Simd::splat(10f32.powf(self.params.gain.unmodulated_plain_value() * (1. / 20.))),
            block_len
        );

        let get_output = self.params.mode.unmodulated_plain_value().output_function::<2>();

        for mut frame in buffer.iter_samples() {

            let sample = array::from_fn(
                |i| *unsafe { frame.get_unchecked_mut(i) }
            ).into();

            self.filter.update_all_smoothers();

            self.filter.process(sample);

            let sample = get_output(&self.filter);

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

        self.filter.set_sample_rate(buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

impl Vst3Plugin for SVFFilter {

    const VST3_CLASS_ID: [u8; 16] = *b"svf_monkeeeeeeee";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Filter,
        Vst3SubCategory::Fx,
    ];
}

impl ClapPlugin for SVFFilter {

    const CLAP_ID: &'static str = "com.AquaEBM.linear_svf";

    const CLAP_DESCRIPTION: Option<&'static str> = Some("Linear SVF Filter");

    const CLAP_MANUAL_URL: Option<&'static str> = None;

    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
    ];
}

nih_export_clap!(SVFFilter);
nih_export_vst3!(SVFFilter);
#![feature(portable_simd)]

use plugin_util::{
    simd_util::map,
    smoothing::{LogSmoother, SIMDSmoother},
    filter::Integrator,
};

use nih_plug::prelude::*;
use core_simd::simd::*;
use std_float::StdFloat;

use std::{sync::Arc, array, f32::consts::PI};

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
pub struct SVF<const N: usize>
where
    LaneCount<N>: SupportedLaneCount
{
    g: LogSmoother<N>,
    r: LogSmoother<N>,
    k: LogSmoother<N>,
    s: [Integrator<N> ; 2],
    pi_tick: Simd<f32, N>,
    x: Simd<f32, N>,
    hp: Simd<f32, N>,
    bp: Simd<f32, N>,
    lp: Simd<f32, N>,
}

impl<const N: usize> SVF<N>
where
    LaneCount<N>: SupportedLaneCount
{
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.pi_tick = Simd::splat(PI / sr);
    }

    pub fn reset(&mut self) {
        self.s[0].reset();
        self.s[1].reset();
    }

    pub fn set_cutoff(&mut self, cutoff: Simd<f32, N>) {
        *self.g = map(cutoff * self.pi_tick, f32::tan);
    }

    pub fn set_cutoff_smoothed(&mut self, cutoff: Simd<f32, N>, block_len: usize) {

        self.g.set_target(map(cutoff * self.pi_tick, f32::tan), block_len)
    }

    pub fn set_resonance(&mut self, res: Simd<f32, N>) {
        *self.r = res;
    }

    pub fn set_resonance_smoothed(&mut self, res: Simd<f32, N>, block_len: usize) {
        self.r.set_target(res, block_len)
    }

    pub fn set_gain(&mut self, k: Simd<f32, N>) {
        *self.k = k;
    }

    pub fn set_gain_smoothed(&mut self, k: Simd<f32, N>, block_len: usize) {
        self.k.set_target(k, block_len)
    }

    pub fn set_params(&mut self, cutoff: Simd<f32, N>, res: Simd<f32, N>, gain: Simd<f32, N>) {
        self.set_cutoff(cutoff);
        self.set_gain(gain);
        self.set_resonance(res);
    }

    pub fn set_params_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        block_len: usize,
    ) {
        self.set_cutoff_smoothed(cutoff, block_len);
        self.set_resonance_smoothed(res, block_len);
        self.set_gain_smoothed(gain, block_len);
    }

    pub fn update_cutoff_smoother(&mut self) {
        self.g.tick()
    }

    pub fn update_resonance_smoother(&mut self) {
        self.r.tick()
    }

    pub fn update_gain_smoother(&mut self) {
        self.k.tick()
    }

    pub fn update_all_smoothers(&mut self) {
        self.update_cutoff_smoother();
        self.update_gain_smoother();
        self.update_resonance_smoother();
    }

    pub fn process(&mut self, sample: Simd<f32, N>) {

        let g = *self.g;

        let g1 = *self.r + g;

        self.x = sample;
        self.hp = (sample - *self.s[1] - *self.s[0] * g1) / (Simd::splat(1.) + g * g1);
        self.bp = self.s[0].process(self.hp, g);
        self.lp = self.s[1].process(self.bp, g);
    }

    pub fn get_highpass(&self) -> Simd<f32, N> {
        self.hp
    }

    pub fn get_bandpass(&self) -> Simd<f32, N> {
        self.bp
    }

    pub fn get_lowpass(&self) -> Simd<f32, N> {
        self.lp
    }

    pub fn get_bandpass1(&self) -> Simd<f32, N> {
        *self.r * self.bp
    }

    pub fn get_allpass(&self) -> Simd<f32, N> {
        Simd::splat(2.).mul_add(self.get_bandpass1(), -self.x)
    }

    pub fn get_notch(&self) -> Simd<f32, N> {
        self.x - self.get_bandpass1()
    }

    pub fn get_peaking(&self) -> Simd<f32, N> {
        self.lp - self.hp
    }

    pub fn get_high_shelf(&self) -> Simd<f32, N> {

        let hp = self.hp;
        self.k.mul_add(hp, self.x - hp)
    }

    pub fn get_band_shelf(&self) -> Simd<f32, N> {

        let bp1 = self.get_bandpass1();
        self.k.mul_add(bp1, self.x - bp1)
    }

    pub fn get_low_shelf(&self) -> Simd<f32, N> {
        let lp = self.lp;
        self.k.mul_add(lp, self.x - lp)
    }

    pub fn output_function(mode: FilterMode) -> fn(&Self) -> Simd<f32, N> {
        match mode {
            FilterMode::AP => Self::get_allpass,
            FilterMode::HP => Self::get_highpass,
            FilterMode::LP => Self::get_lowpass,
            FilterMode::BP => Self::get_bandpass,
            FilterMode::BP1 => Self::get_bandpass1,
            FilterMode::NCH => Self::get_notch,
            FilterMode::HSH => Self::get_high_shelf,
            FilterMode::BSH => Self::get_band_shelf,
            FilterMode::LSH => Self::get_low_shelf,
            FilterMode::PK => Self::get_peaking,
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

        let get_output = SVF::<2>::output_function(
            self.params.mode.unmodulated_plain_value()
        );

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
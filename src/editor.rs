use core::sync::atomic::Ordering;

use nih_plug::params::Param;
use nih_plug_vizia::vizia::{prelude::*, vg};
use num::Complex;
use plugin_util::{
    simd::f32x1,
    smoothing::{LogSmoother, Smoother},
};

use crate::{Arc, SVFParams, BASE_SAMPLE_RATE, MAX_FREQ, MIN_FREQ, TAU, Filter};

pub struct SVFBode {
    pub params: Arc<SVFParams>,
}

impl SVFBode {
    pub fn new(cx: &mut Context, params: Arc<SVFParams>) -> Handle<Self> {
        SVFBode { params }.build(cx, |_| ())
    }
}

impl View for SVFBode {
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        let width = bounds.width();

        // draw background

        let mut bg = vg::Path::new();

        bg.rect(bounds.x, bounds.y, bounds.w, bounds.h);

        canvas.fill_path(&mut bg, &vg::Paint::color(vg::Color::black()));

        // ----

        const NUM_POINTS: usize = 700;

        let delta_x = bounds.width() / NUM_POINTS as f32;

        let mut smoother = LogSmoother::<1>::default();
        smoother.set_instantly(f32x1::from_array([MIN_FREQ]));
        smoother.set_target(f32x1::from_array([MAX_FREQ]), NUM_POINTS);

        let (mut x, y) = bounds.center_left();

        let mut plot = vg::Path::new();

        plot.move_to(x, y);

        let two_pi_tick = self.params.two_pi_tick.load(Ordering::Relaxed);

        let cutoff_norm = self.params.cutoff.unmodulated_normalized_value();
        let cutoff_freq = MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(cutoff_norm);

        let max_freq = f32::min(
            TAU / BASE_SAMPLE_RATE * MAX_FREQ,
            two_pi_tick * MAX_FREQ
        ) / two_pi_tick;

        let mut freq = smoother.get_current()[0];

        let cutoff_freq = f32::tan(cutoff_freq * two_pi_tick * 0.5);

        let h = Filter::get_transfer_function::<f32>(
            self.params.mode.unmodulated_plain_value()
        );

        let res = self.params.res.unmodulated_plain_value();
        let gain_normalized = self.params.gain.modulated_plain_value();
        let gain = 10f32.powf(gain_normalized * (1. / 20.));

        let mut point_idx = 0;
        while freq < max_freq {
            let w = f32::tan(freq * two_pi_tick * 0.5) / cutoff_freq;

            let impedence = h(Complex::new(0., w), res, gain);

            let gain_db = 10. * f32::log10(impedence.norm_sqr());
            let offset = (gain_db / 35.) * bounds.height() / 2.;

            if point_idx == 0 {
                plot.move_to(x, y - offset);
                point_idx = 1;
            } else {
                plot.line_to(x, y - offset);
            }

            x += delta_x;

            smoother.tick();
            freq = smoother.get_current()[0];
        }

        let paint = vg::Paint::linear_gradient(
            0.,
            0.,
            width,
            0.,
            vg::Color::rgbf(1., 0., 0.),
            vg::Color::rgbf(0., 1., 0.),
        )
        .with_miter_limit(0.)
        .with_line_width(3.)
        .with_anti_alias(true)
        .with_stencil_strokes(true);

        canvas.stroke_path(&mut plot, &paint);
    }
}

use svf::SVF;
use nih_plug::prelude::nih_export_standalone;

fn main() {
    nih_export_standalone::<SVF<2>>();
}
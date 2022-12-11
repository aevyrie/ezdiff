use ezdiff::*;
use num_traits::Pow;

pub fn main() {
    let x = dual!(5.0);
    let y = (x.pow(2.0)).cos();
    dbg!(y);

    // f(x) = cos(x^2) + 3x
    let f = |x: Dual<f32>| (x.pow(2.0)).cos() + 3.0 * x;
    // evaluate at x = 2
    dbg!(f(dual!(2.0)));

    let f = |x: Dual<f32>| (x.pow(3.0)).pow(1.0 / 3.0);
    dbg!(f(dual!(0.0)));
}

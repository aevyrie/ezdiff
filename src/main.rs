use ezdiff::*;
use num_traits::Pow;

pub fn main() {
    let x = dual!(5.0);
    let y = (x.pow(2.0)).cos();
    dbg!(y);

    // f(x) = cos(x^2) + 3x
    let f = |x: f32| (dual!(x).pow(2.0)).cos() + 3.0 * dual!(x);
    // evaluate at x = 2
    dbg!(f(2.0));
}

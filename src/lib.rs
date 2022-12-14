use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use num_traits::{Float, Pow};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Dual<F: Float> {
    x: F,
    dx: F,
}

impl<F: Float> Dual<F> {
    #[inline]
    pub fn new(val: F) -> Self {
        Self {
            x: val,
            dx: F::one(),
        }
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        self.pow(F::from(0.5).unwrap())
    }

    #[inline]
    pub fn exp(self) -> Self {
        Dual {
            x: self.x.exp(),
            dx: self.x.exp() * self.dx,
        }
    }

    #[inline]
    pub fn ln(self) -> Self {
        Dual {
            x: self.x.ln(),
            dx: self.x.powi(-1) * self.dx,
        }
    }

    #[inline]
    pub fn log(self, base: F) -> Self {
        Dual {
            x: self.x.log(base),
            dx: (base.ln() * self.x).powi(-1) * self.dx,
        }
    }

    #[inline]
    pub fn sin(self) -> Self {
        Dual {
            x: self.x.sin(),
            dx: self.x.cos() * self.dx,
        }
    }

    #[inline]
    pub fn cos(self) -> Self {
        Dual {
            x: self.x.cos(),
            dx: -self.x.sin() * self.dx,
        }
    }

    #[inline]
    pub fn tan(self) -> Self {
        Dual {
            x: self.x.tan(),
            dx: self.x.cos().powi(-2) * self.dx,
        }
    }

    #[inline]
    pub fn asin(self) -> Self {
        Dual {
            x: self.x.asin(),
            dx: (F::one() - self.x.powi(2)).sqrt().powi(-1) * self.dx,
        }
    }

    #[inline]
    pub fn acos(self) -> Self {
        Dual {
            x: self.x.acos(),
            dx: -(F::one() - self.x.powi(2)).sqrt().powi(-1) * self.dx,
        }
    }

    #[inline]
    pub fn atan(self) -> Self {
        Dual {
            x: self.x.atan(),
            dx: (F::one() + self.x.powi(2)).powi(-1) * self.dx,
        }
    }

    pub fn value(&self) -> F {
        self.x
    }

    pub fn derivative(&self) -> F {
        self.dx
    }
}

impl<F: Float> Neg for Dual<F> {
    type Output = Dual<F>;

    fn neg(self) -> Self::Output {
        Dual {
            x: self.x.neg(),
            dx: self.dx.neg(),
        }
    }
}

// Sum rule
impl<F: Float> Add for Dual<F> {
    type Output = Dual<F>;

    fn add(self, rhs: Self) -> Self::Output {
        Dual {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx,
        }
    }
}

// Sum constant
impl<F: Float> Add<F> for Dual<F> {
    type Output = Dual<F>;

    fn add(self, rhs: F) -> Self::Output {
        Dual {
            x: self.x + rhs,
            dx: self.dx,
        }
    }
}

// Sum constant
impl Add<Dual<f32>> for f32 {
    type Output = Dual<f32>;

    fn add(self, rhs: Dual<f32>) -> Self::Output {
        Dual {
            x: rhs.x + self,
            dx: rhs.dx,
        }
    }
}

// Sum constant
impl Add<Dual<f64>> for f64 {
    type Output = Dual<f64>;

    fn add(self, rhs: Dual<f64>) -> Self::Output {
        Dual {
            x: rhs.x + self,
            dx: rhs.dx,
        }
    }
}

// Sum constant
impl<F: Float> Add<Dual<F>> for (F,) {
    type Output = Dual<F>;

    fn add(self, rhs: Dual<F>) -> Self::Output {
        Dual {
            x: rhs.x + self.0,
            dx: rhs.dx,
        }
    }
}

// Difference rule
impl<F: Float> Sub for Dual<F> {
    type Output = Dual<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        Dual {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx,
        }
    }
}

// Product rule
impl<F: Float> Mul for Dual<F> {
    type Output = Dual<F>;

    fn mul(self, rhs: Dual<F>) -> Self::Output {
        Dual {
            x: self.x * rhs.x,
            dx: self.x * rhs.dx + rhs.x * self.dx,
        }
    }
}

// Constant multiple rule
impl<F: Float> Mul<F> for Dual<F> {
    type Output = Dual<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Dual {
            x: self.x * rhs,
            dx: self.dx * rhs,
        }
    }
}

// Constant multiple rule
impl Mul<Dual<f32>> for f32 {
    type Output = Dual<f32>;

    fn mul(self, rhs: Dual<f32>) -> Self::Output {
        Dual {
            x: self * rhs.x,
            dx: self * rhs.dx,
        }
    }
}

// Constant multiple rule
impl Mul<Dual<f64>> for f64 {
    type Output = Dual<f64>;

    fn mul(self, rhs: Dual<f64>) -> Self::Output {
        Dual {
            x: self * rhs.x,
            dx: self * rhs.dx,
        }
    }
}

// Quotient rule
impl<F: Float> Div for Dual<F> {
    type Output = Dual<F>;

    fn div(self, rhs: Dual<F>) -> Self::Output {
        Dual {
            x: self.x / rhs.x,
            dx: (self.x * rhs.dx + rhs.x * self.dx) / (rhs.x * rhs.x),
        }
    }
}

// Power rule
impl<F: Float> Pow<F> for Dual<F> {
    type Output = Dual<F>;

    fn pow(self, rhs: F) -> Self::Output {
        Dual {
            x: self.x.powf(rhs),
            dx: rhs * self.x.powf(rhs - F::one()) * self.dx, // n * x^(n-1) * d/dx
        }
    }
}

// Inverse(?) power rule a^x
impl<F: Float> Pow<Dual<F>> for (F,) {
    type Output = Dual<F>;

    fn pow(self, rhs: Dual<F>) -> Self::Output {
        Dual {
            x: self.0.powf(rhs.x),
            dx: self.0.ln() * self.0.powf(rhs.x) * rhs.dx,
        }
    }
}

// Inverse(?) power rule a^x
impl Pow<Dual<f32>> for f32 {
    type Output = Dual<f32>;

    fn pow(self, rhs: Dual<f32>) -> Self::Output {
        Dual {
            x: self.powf(rhs.x),
            dx: self.ln() * self.powf(rhs.x) * rhs.dx,
        }
    }
}

// Inverse(?) power rule a^x
impl Pow<Dual<f64>> for f64 {
    type Output = Dual<f64>;

    fn pow(self, rhs: Dual<f64>) -> Self::Output {
        Dual {
            x: self.powf(rhs.x),
            dx: self.ln() * self.powf(rhs.x) * rhs.dx,
        }
    }
}

#[macro_export]
macro_rules! dual {
    ($a:expr) => {{
        Dual::new($a)
    }};
}

impl From<f32> for Dual<f32> {
    #[inline]
    fn from(input: f32) -> Self {
        Dual::new(input)
    }
}

impl From<f64> for Dual<f64> {
    #[inline]
    fn from(input: f64) -> Self {
        Dual::new(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let x = dual!(3.0);
        let y = x * x + 2.0;
        assert_eq!(y.x, 11.0);
        assert_eq!(y.dx, 6.0);
    }

    #[test]
    fn sin() {
        let sin = |x: Dual<_>| x.sin();
        let y_1 = sin(dual!(1.0));
        assert_eq!(y_1.x, 0.8414709848078965);
        assert_eq!(y_1.dx, 0.5403023058681398);
    }

    #[test]
    fn cos() {
        let cos = |x: Dual<_>| x.cos();
        let y_1 = cos(dual!(1.0));
        assert_eq!(y_1.x, 0.5403023058681398);
        assert_eq!(y_1.dx, -0.8414709848078965);
    }

    #[test]
    fn tan() {
        let tan = |x: Dual<_>| x.tan();
        let y_1 = tan(dual!(1.0));
        assert_eq!(y_1.x, 1.5574077246549023);
        assert_eq!(y_1.dx, 3.425518820814759);
    }

    #[test]
    fn asin() {
        let asin = |x: Dual<_>| x.asin();
        let y_05 = asin(dual!(0.5));
        assert_eq!(y_05.x, 0.5235987755982989);
        assert_eq!(y_05.dx, 1.1547005383792517);
    }

    #[test]
    fn acos() {
        let acos = |x: Dual<_>| x.acos();
        let y_05 = acos(dual!(0.5));
        assert_eq!(y_05.x, 1.0471975511965979);
        assert_eq!(y_05.dx, -1.1547005383792517);
    }

    #[test]
    fn atan() {
        let atan = |x: Dual<_>| x.atan();
        let y_05 = atan(dual!(0.5));
        assert_eq!(y_05.x, 0.4636476090008061);
        assert_eq!(y_05.dx, 0.8);
    }

    #[test]
    fn sqrt() {
        let sqrt = |x: Dual<_>| x.sqrt();
        let y_1 = sqrt(dual!(1.0));
        assert_eq!(y_1.x, 1.0);
        assert_eq!(y_1.dx, 0.5);
    }

    #[test]
    fn exp() {
        let exp = |x: Dual<_>| x.exp();
        let y_1 = exp(dual!(1.0));
        assert_eq!(y_1.x, std::f32::consts::E);
        assert_eq!(y_1.dx, std::f32::consts::E);
    }

    #[test]
    fn ln() {
        let ln = |x: Dual<_>| x.ln();
        let y_2 = ln(dual!(2.0));
        assert_eq!(y_2.x, 0.6931471805599453);
        assert_eq!(y_2.dx, 0.5);
    }

    #[test]
    fn log() {
        let log = |x: Dual<_>| x.log(10.0);
        let y_2 = log(dual!(2.0));
        assert_eq!(y_2.x, 0.30102999566398114);
        assert_eq!(y_2.dx, 0.21714724095162588);
    }

    #[test]
    fn add_mul_consts() {
        let f = |x: Dual<f32>| 1.0 + x * 3.0;
        let y_2 = f(dual!(2.0));
        assert_eq!(y_2.x, 7.0);
        assert_eq!(y_2.dx, 3.0);
    }

    #[test]
    fn product() {
        let f = |x: Dual<f32>| x.sin() * x.cos();
        let y_1 = f(dual!(1.0));
        assert_eq!(y_1.x, 0.45464867);
        assert_eq!(y_1.dx, -0.4161468);
    }
}

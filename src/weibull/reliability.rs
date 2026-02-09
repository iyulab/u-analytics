//! Reliability analysis from fitted Weibull parameters.
//!
//! Provides reliability function, hazard rate, MTBF, B-life, and other
//! common reliability engineering metrics.

use u_numerics::special::gamma;

/// Reliability analysis results from a fitted Weibull distribution.
///
/// Computes reliability engineering metrics (reliability function, hazard
/// rate, MTBF, B-life) from Weibull shape (beta) and scale (eta) parameters.
///
/// # Mathematical Background
///
/// Given a Weibull distribution with shape beta > 0 and scale eta > 0:
/// - Reliability: R(t) = exp(-(t/eta)^beta)
/// - Hazard rate: lambda(t) = (beta/eta) * (t/eta)^(beta-1)
/// - MTBF: eta * Gamma(1 + 1/beta)
///
/// # Examples
///
/// ```
/// use u_analytics::weibull::ReliabilityAnalysis;
/// let ra = ReliabilityAnalysis::new(2.0, 100.0).unwrap();
/// assert!((ra.reliability(0.0) - 1.0).abs() < 1e-10);
/// assert!(ra.hazard_rate(50.0) > 0.0);
/// assert!(ra.mtbf() > 0.0);
/// let b10 = ra.b_life(0.10).unwrap();
/// assert!(b10 > 0.0 && b10 < 100.0);
/// ```
///
/// # Reference
/// Meeker & Escobar (1998), *Statistical Methods for Reliability Data*, Wiley.
#[derive(Debug, Clone)]
pub struct ReliabilityAnalysis {
    /// Shape parameter (beta).
    shape: f64,
    /// Scale parameter (eta).
    scale: f64,
}

impl ReliabilityAnalysis {
    /// Creates a new reliability analysis from Weibull parameters.
    ///
    /// # Arguments
    /// * `shape` - Shape parameter beta (must be positive and finite)
    /// * `scale` - Scale parameter eta (must be positive and finite)
    ///
    /// # Returns
    /// `None` if either parameter is non-positive or non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::weibull::ReliabilityAnalysis;
    /// let ra = ReliabilityAnalysis::new(2.0, 100.0);
    /// assert!(ra.is_some());
    ///
    /// // Invalid parameters return None
    /// assert!(ReliabilityAnalysis::new(-1.0, 100.0).is_none());
    /// assert!(ReliabilityAnalysis::new(2.0, 0.0).is_none());
    /// ```
    pub fn new(shape: f64, scale: f64) -> Option<Self> {
        if !shape.is_finite() || !scale.is_finite() || shape <= 0.0 || scale <= 0.0 {
            return None;
        }
        Some(Self { shape, scale })
    }

    /// Creates a reliability analysis from an MLE fitting result.
    ///
    /// # Panics
    /// This function does not panic. MLE results always have valid parameters.
    pub fn from_mle(result: &super::mle::WeibullMleResult) -> Self {
        Self {
            shape: result.shape,
            scale: result.scale,
        }
    }

    /// Creates a reliability analysis from an MRR fitting result.
    ///
    /// # Panics
    /// This function does not panic. MRR results always have valid parameters.
    pub fn from_mrr(result: &super::mrr::WeibullMrrResult) -> Self {
        Self {
            shape: result.shape,
            scale: result.scale,
        }
    }

    /// Returns the shape parameter (beta).
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns the scale parameter (eta).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Reliability (survival) function at time t.
    ///
    /// ```text
    /// R(t) = exp(-(t/eta)^beta)
    /// ```
    ///
    /// For t < 0, returns 1.0 (no failure before time 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::weibull::ReliabilityAnalysis;
    /// let ra = ReliabilityAnalysis::new(2.0, 100.0).unwrap();
    ///
    /// // R(0) = 1.0 (full reliability at time zero)
    /// assert!((ra.reliability(0.0) - 1.0).abs() < 1e-10);
    ///
    /// // R(eta) = exp(-1) ≈ 0.368 for any shape parameter
    /// let expected = (-1.0_f64).exp();
    /// assert!((ra.reliability(100.0) - expected).abs() < 1e-10);
    /// ```
    ///
    /// # Reference
    /// Weibull (1951), *Journal of Applied Mechanics* 18(3), pp. 293-297.
    pub fn reliability(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let z = t / self.scale;
        (-z.powf(self.shape)).exp()
    }

    /// Failure rate (hazard function) at time t.
    ///
    /// ```text
    /// lambda(t) = (beta/eta) * (t/eta)^(beta-1)
    /// ```
    ///
    /// - beta < 1: Decreasing failure rate (infant mortality)
    /// - beta = 1: Constant failure rate (random/exponential failures)
    /// - beta > 1: Increasing failure rate (wear-out)
    ///
    /// For t <= 0, returns 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::weibull::ReliabilityAnalysis;
    /// let ra = ReliabilityAnalysis::new(2.0, 100.0).unwrap();
    ///
    /// // Hazard rate is positive for t > 0
    /// assert!(ra.hazard_rate(50.0) > 0.0);
    ///
    /// // With beta > 1, hazard rate increases over time (wear-out)
    /// assert!(ra.hazard_rate(50.0) < ra.hazard_rate(80.0));
    /// ```
    ///
    /// # Reference
    /// Meeker & Escobar (1998), *Statistical Methods for Reliability Data*, Ch. 4.
    pub fn hazard_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let z = t / self.scale;
        (self.shape / self.scale) * z.powf(self.shape - 1.0)
    }

    /// Mean Time Between Failures (MTBF).
    ///
    /// ```text
    /// MTBF = eta * Gamma(1 + 1/beta)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::weibull::ReliabilityAnalysis;
    /// let ra = ReliabilityAnalysis::new(2.0, 100.0).unwrap();
    /// let mtbf = ra.mtbf();
    /// assert!(mtbf > 0.0);
    ///
    /// // For beta=1 (exponential), MTBF = eta
    /// let exp_ra = ReliabilityAnalysis::new(1.0, 50.0).unwrap();
    /// assert!((exp_ra.mtbf() - 50.0).abs() < 1e-8);
    /// ```
    ///
    /// # Reference
    /// Johnson, Kotz & Balakrishnan (1994), *Continuous Univariate Distributions*,
    /// Vol. 1, Chapter 21.
    pub fn mtbf(&self) -> f64 {
        self.scale * gamma(1.0 + 1.0 / self.shape)
    }

    /// Time at which reliability drops to a given level.
    ///
    /// Solves R(t) = p for t:
    ///
    /// ```text
    /// t = eta * (-ln(p))^(1/beta)
    /// ```
    ///
    /// # Arguments
    /// * `p` - Desired reliability level, must be in (0, 1).
    ///
    /// # Returns
    /// `None` if `p` is outside (0, 1).
    ///
    /// # Reference
    /// Abernethy (2006), *The New Weibull Handbook*, 5th ed.
    pub fn time_to_reliability(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        Some(self.scale * (-p.ln()).powf(1.0 / self.shape))
    }

    /// B-life: time at which a given fraction of the population has failed.
    ///
    /// B10 life (10% failed) = `b_life(0.10)`, which is equivalent to
    /// `time_to_reliability(0.90)`.
    ///
    /// # Arguments
    /// * `fraction_failed` - Fraction of population that has failed, must be in (0, 1).
    ///
    /// # Returns
    /// `None` if `fraction_failed` is outside (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::weibull::ReliabilityAnalysis;
    /// let ra = ReliabilityAnalysis::new(2.0, 100.0).unwrap();
    ///
    /// // B10: time when 10% of the population has failed
    /// let b10 = ra.b_life(0.10).unwrap();
    /// assert!(b10 > 0.0 && b10 < 100.0);
    ///
    /// // B-lives increase with failure fraction: B5 < B10 < B50
    /// let b5 = ra.b_life(0.05).unwrap();
    /// let b50 = ra.b_life(0.50).unwrap();
    /// assert!(b5 < b10 && b10 < b50);
    /// ```
    ///
    /// # Reference
    /// Abernethy (2006), *The New Weibull Handbook*, 5th ed., Chapter 2.
    pub fn b_life(&self, fraction_failed: f64) -> Option<f64> {
        if fraction_failed <= 0.0 || fraction_failed >= 1.0 {
            return None;
        }
        self.time_to_reliability(1.0 - fraction_failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weibull::{weibull_mle, weibull_mrr};

    #[test]
    fn test_new_valid() {
        assert!(ReliabilityAnalysis::new(2.0, 50.0).is_some());
    }

    #[test]
    fn test_new_invalid() {
        assert!(ReliabilityAnalysis::new(0.0, 50.0).is_none());
        assert!(ReliabilityAnalysis::new(-1.0, 50.0).is_none());
        assert!(ReliabilityAnalysis::new(2.0, 0.0).is_none());
        assert!(ReliabilityAnalysis::new(2.0, -1.0).is_none());
        assert!(ReliabilityAnalysis::new(f64::NAN, 50.0).is_none());
        assert!(ReliabilityAnalysis::new(2.0, f64::INFINITY).is_none());
    }

    #[test]
    fn test_reliability_at_zero() {
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        assert!((ra.reliability(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_reliability_decreasing() {
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        let mut prev = 1.0;
        for i in 1..=100 {
            let t = i as f64;
            let r = ra.reliability(t);
            assert!(
                r <= prev + 1e-15,
                "reliability should be non-increasing: R({}) = {} > R({}) = {}",
                t,
                r,
                t - 1.0,
                prev
            );
            prev = r;
        }
    }

    #[test]
    fn test_reliability_at_scale() {
        // R(eta) = exp(-1) for any beta
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        let r_at_scale = ra.reliability(50.0);
        let expected = (-1.0_f64).exp();
        assert!(
            (r_at_scale - expected).abs() < 1e-10,
            "R(eta) = {}, expected exp(-1) = {}",
            r_at_scale,
            expected
        );
    }

    #[test]
    fn test_reliability_negative_time() {
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        assert!((ra.reliability(-10.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_hazard_rate_exponential() {
        // beta=1 => constant hazard rate = 1/eta
        let ra = ReliabilityAnalysis::new(1.0, 20.0).expect("valid parameters");
        for t in [5.0, 10.0, 20.0, 50.0] {
            assert!(
                (ra.hazard_rate(t) - 1.0 / 20.0).abs() < 1e-10,
                "hazard at t={} should be 1/eta=0.05",
                t
            );
        }
    }

    #[test]
    fn test_hazard_rate_increasing() {
        // beta > 1 => increasing hazard rate (wear-out)
        let ra = ReliabilityAnalysis::new(3.0, 50.0).expect("valid parameters");
        let h1 = ra.hazard_rate(10.0);
        let h2 = ra.hazard_rate(20.0);
        let h3 = ra.hazard_rate(30.0);
        assert!(h1 < h2, "hazard should increase: h(10)={} >= h(20)={}", h1, h2);
        assert!(h2 < h3, "hazard should increase: h(20)={} >= h(30)={}", h2, h3);
    }

    #[test]
    fn test_hazard_rate_zero_time() {
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        assert!((ra.hazard_rate(0.0)).abs() < 1e-15);
        assert!((ra.hazard_rate(-5.0)).abs() < 1e-15);
    }

    #[test]
    fn test_mtbf_exponential() {
        // beta=1 => MTBF = eta * Gamma(2) = eta * 1 = eta
        let ra = ReliabilityAnalysis::new(1.0, 100.0).expect("valid parameters");
        assert!(
            (ra.mtbf() - 100.0).abs() < 1e-8,
            "MTBF = {}, expected 100.0 for exponential",
            ra.mtbf()
        );
    }

    #[test]
    fn test_mtbf_rayleigh() {
        // beta=2, eta=1 => MTBF = Gamma(1.5) = sqrt(pi)/2
        let ra = ReliabilityAnalysis::new(2.0, 1.0).expect("valid parameters");
        let expected = std::f64::consts::PI.sqrt() / 2.0;
        assert!(
            (ra.mtbf() - expected).abs() < 1e-10,
            "MTBF = {}, expected sqrt(pi)/2 = {}",
            ra.mtbf(),
            expected
        );
    }

    #[test]
    fn test_time_to_reliability_median() {
        // Median life: R(t) = 0.5 => t = eta * (ln(2))^(1/beta)
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        let t_median = ra
            .time_to_reliability(0.5)
            .expect("p=0.5 is valid");
        let expected = 50.0 * (2.0_f64.ln()).powf(0.5);
        assert!(
            (t_median - expected).abs() < 1e-10,
            "median life = {}, expected {}",
            t_median,
            expected
        );
    }

    #[test]
    fn test_time_to_reliability_invalid() {
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        assert!(ra.time_to_reliability(0.0).is_none());
        assert!(ra.time_to_reliability(1.0).is_none());
        assert!(ra.time_to_reliability(-0.1).is_none());
        assert!(ra.time_to_reliability(1.5).is_none());
    }

    #[test]
    fn test_time_to_reliability_roundtrip() {
        let ra = ReliabilityAnalysis::new(2.5, 100.0).expect("valid parameters");
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let t = ra.time_to_reliability(p).expect("valid p");
            let r = ra.reliability(t);
            assert!(
                (r - p).abs() < 1e-10,
                "roundtrip: time_to_reliability({}) = {}, reliability({}) = {}",
                p,
                t,
                t,
                r
            );
        }
    }

    #[test]
    fn test_b_life_b10() {
        // B10 = time when 10% have failed = time_to_reliability(0.90)
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        let b10 = ra.b_life(0.10).expect("valid fraction");
        let t_r90 = ra.time_to_reliability(0.90).expect("valid p");
        assert!(
            (b10 - t_r90).abs() < 1e-10,
            "B10 = {}, time_to_reliability(0.90) = {}",
            b10,
            t_r90
        );
    }

    #[test]
    fn test_b_life_invalid() {
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        assert!(ra.b_life(0.0).is_none());
        assert!(ra.b_life(1.0).is_none());
        assert!(ra.b_life(-0.1).is_none());
    }

    #[test]
    fn test_from_mle() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let mle = weibull_mle(&data).expect("MLE should converge");
        let ra = ReliabilityAnalysis::from_mle(&mle);

        assert!((ra.shape() - mle.shape).abs() < 1e-15);
        assert!((ra.scale() - mle.scale).abs() < 1e-15);
        assert!(ra.mtbf() > 0.0);
    }

    #[test]
    fn test_from_mrr() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let mrr = weibull_mrr(&data).expect("MRR should succeed");
        let ra = ReliabilityAnalysis::from_mrr(&mrr);

        assert!((ra.shape() - mrr.shape).abs() < 1e-15);
        assert!((ra.scale() - mrr.scale).abs() < 1e-15);
        assert!(ra.mtbf() > 0.0);
    }

    #[test]
    fn test_b_life_ordering() {
        // B5 < B10 < B50 (more failures => more time)
        let ra = ReliabilityAnalysis::new(2.0, 50.0).expect("valid parameters");
        let b5 = ra.b_life(0.05).expect("valid");
        let b10 = ra.b_life(0.10).expect("valid");
        let b50 = ra.b_life(0.50).expect("valid");
        assert!(b5 < b10, "B5={} should < B10={}", b5, b10);
        assert!(b10 < b50, "B10={} should < B50={}", b10, b50);
    }

    #[test]
    fn test_hazard_pdf_reliability_relation() {
        // h(t) = f(t) / R(t) where f(t) is the Weibull PDF
        // f(t) = (beta/eta) * (t/eta)^(beta-1) * exp(-(t/eta)^beta)
        let ra = ReliabilityAnalysis::new(2.5, 50.0).expect("valid parameters");
        for t in [5.0, 20.0, 50.0, 80.0] {
            let z = t / ra.scale();
            let pdf = (ra.shape() / ra.scale()) * z.powf(ra.shape() - 1.0) * (-z.powf(ra.shape())).exp();
            let h = ra.hazard_rate(t);
            let r = ra.reliability(t);
            assert!(
                (h * r - pdf).abs() < 1e-10,
                "h(t)*R(t) = {} should equal f(t) = {} at t={}",
                h * r,
                pdf,
                t
            );
        }
    }

    #[test]
    fn test_accessors() {
        let ra = ReliabilityAnalysis::new(2.5, 100.0).expect("valid parameters");
        assert!((ra.shape() - 2.5).abs() < 1e-15);
        assert!((ra.scale() - 100.0).abs() < 1e-15);
    }
}

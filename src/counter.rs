//! Increments of a cumulative counter, each classified by what explains it.
//!
//! A cumulative counter -- an energy, gas or water meter's register, an
//! hour meter, a production or byte counter -- is read at increasing times,
//! and what is wanted is how much it advanced between readings. A reading
//! below the one before it has more than one explanation: the register
//! passed its maximum and started again from 0 (a *wrap*), it was reset or
//! replaced, or the reading is wrong. These functions do not pick one by
//! assumption. Each pair of neighbouring readings is classified:
//!
//! - [`IncrementKind::Normal`] -- the value did not fall; the increase is the
//!   difference.
//! - [`IncrementKind::Declared`] -- the caller declared a reset or a
//!   replacement inside the pair ([`Discontinuity`]); the increase is counted
//!   from the new start value (plus, when the old register's last value is
//!   given, what it advanced before the change).
//! - [`IncrementKind::Wrap`] -- the value fell and a known `modulus` (the
//!   value at which the register returns to 0) explains it: the increase is
//!   `modulus − previous + next`. With `max_rate`, a wrap is accepted only
//!   when that increase is one the counter can make in the time between the
//!   readings.
//! - [`IncrementKind::Unexplained`] -- the value fell and nothing given
//!   explains it: the increase is `None`, not a guess.
//!
//! Monitoring systems usually read every fall as a reset and count from 0
//! (Prometheus' `increase()` and `rate()`); that is right for process
//! counters that restart with their process, and wrong for a meter whose
//! readings settle a bill. Classifying, rather than assuming, leaves the
//! choice to the caller, who can see what was not explained.
//!
//! [`counter_totals`] sums the increments over consecutive intervals (a day,
//! a quarter of an hour), with the number of pairs left unexplained in each.
//!
//! # References
//!
//! - Prometheus documentation, "Query functions": `increase()`, `rate()` --
//!   counter resets are compensated by assuming a restart from 0.
//! - IEC 62056-6-2 (DLMS/COSEM), register objects with a scaler and a
//!   wrap-around value -- the `modulus` here.

/// A reset or replacement of the counter, declared by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Discontinuity {
    /// When it happened. It belongs to the pair of readings around it: after
    /// the earlier reading, at or before the later one.
    pub time: f64,
    /// The register's value right after it -- 0 for a reset to zero, the new
    /// register's reading for a replacement.
    pub start: f64,
    /// The old register's last value, when it was read: the increase then
    /// also counts what the counter advanced from the earlier reading up to
    /// the change. `None`: that part is unknown and not counted.
    pub last: Option<f64>,
}

/// What explains an increment (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncrementKind {
    /// The value did not fall.
    Normal,
    /// A declared reset or replacement lies inside the pair.
    Declared,
    /// The value fell and the register's `modulus` explains it.
    Wrap,
    /// The value fell and nothing given explains it.
    Unexplained,
}

/// How much the counter advanced between two neighbouring readings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CounterIncrement {
    /// The earlier reading's time.
    pub start: f64,
    /// The later reading's time.
    pub end: f64,
    /// The increase; `None` when the pair is [`IncrementKind::Unexplained`].
    pub increase: Option<f64>,
    /// What explains it.
    pub kind: IncrementKind,
}

/// The increments that end in one interval `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CounterTotal {
    /// The interval's start (included).
    pub start: f64,
    /// The interval's end (excluded).
    pub end: f64,
    /// The sum of the known increases of the pairs ending in the interval.
    /// With `unexplained > 0` it leaves those pairs out, so it is a lower
    /// bound of the true advance only if their advance was not negative.
    pub increase: f64,
    /// The pairs ending in the interval.
    pub pairs: usize,
    /// How many of them are [`IncrementKind::Unexplained`].
    pub unexplained: usize,
}

/// Why readings, settings or intervals were refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CounterError {
    /// `times` and `values` differ in length.
    LengthMismatch {
        /// The number of times.
        times: usize,
        /// The number of values.
        values: usize,
    },
    /// Fewer than two readings: there is no pair.
    TooFewReadings {
        /// The number given.
        got: usize,
    },
    /// A NaN or an infinity. `parameter` names the input (`times`, `values`,
    /// `discontinuities.time`, ...), `index` its position.
    NotFinite {
        /// The input.
        parameter: &'static str,
        /// The position in it; `None` for a single setting.
        index: Option<usize>,
    },
    /// A time not after the one before it (`times`, or interval `edges`).
    Unordered {
        /// The input.
        parameter: &'static str,
        /// The position of the time that is not after its predecessor.
        index: usize,
    },
    /// `modulus` or `max_rate` not above 0.
    NotPositive {
        /// The setting.
        parameter: &'static str,
        /// The value given.
        got: f64,
    },
    /// A value outside `[0, modulus)`: a register with that wrap value never
    /// shows it.
    OutsideModulus {
        /// The input (`values`, `discontinuities.start`, `discontinuities.last`).
        parameter: &'static str,
        /// The position.
        index: usize,
        /// The value given.
        got: f64,
        /// The modulus.
        modulus: f64,
    },
    /// A discontinuity whose time lies in no pair: at or before the first
    /// reading, or after the last.
    OutsideReadings {
        /// The discontinuity's position.
        index: usize,
        /// Its time.
        got: f64,
        /// The first reading's time: a discontinuity must be after it.
        first: f64,
        /// The last reading's time: a discontinuity must be at or before it.
        last: f64,
    },
    /// Two discontinuities in one pair of readings: what the counter showed
    /// between them was not read.
    TwoInOnePair {
        /// The second one's position.
        index: usize,
    },
}

impl std::fmt::Display for CounterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::LengthMismatch { times, values } => {
                write!(f, "{times} times but {values} values; each reading needs both")
            }
            Self::TooFewReadings { got } => {
                write!(f, "at least 2 readings are needed, got {got}")
            }
            Self::NotFinite {
                parameter,
                index: Some(i),
            } => write!(f, "{parameter}[{i}] is not a finite number"),
            Self::NotFinite {
                parameter,
                index: None,
            } => write!(f, "{parameter} is not a finite number"),
            Self::Unordered { parameter, index } => write!(
                f,
                "{parameter}[{index}] is not after {parameter}[{}]; times must increase",
                index - 1
            ),
            Self::NotPositive { parameter, got } => {
                write!(f, "{parameter} must be > 0, got {got}")
            }
            Self::OutsideModulus {
                parameter,
                index,
                got,
                modulus,
            } => write!(
                f,
                "{parameter}[{index}] is {got}; a register that wraps at {modulus} shows values in [0, {modulus})"
            ),
            Self::OutsideReadings {
                index,
                got,
                first,
                last,
            } => write!(
                f,
                "discontinuities[{index}] is at {got}, outside every pair of readings: \
                 it must be after {first} and at or before {last}"
            ),
            Self::TwoInOnePair { index } => write!(
                f,
                "discontinuities[{index}] falls in the same pair of readings as the one \
                 before it; what the counter showed between them was not read"
            ),
        }
    }
}

impl std::error::Error for CounterError {}

/// The settings of [`counter_increments`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CounterOptions {
    /// The value at which the register returns to 0 (`1_000_000` for a
    /// six-digit register): a fall is then a wrap. `None`: a fall is a wrap
    /// only if a discontinuity is declared, otherwise unexplained.
    pub modulus: Option<f64>,
    /// The most the counter can advance per unit of time. A wrap whose
    /// increase is more than `max_rate × (end − start)` is unexplained rather
    /// than a wrap -- a reset near the top of the register looks like a wrap
    /// otherwise.
    pub max_rate: Option<f64>,
    /// Resets and replacements the caller knows of, in time order.
    pub discontinuities: Vec<Discontinuity>,
}

/// The increment between each pair of neighbouring readings, classified.
///
/// `times` must increase strictly; `values` are the register's readings at
/// those times. The result has one entry per pair, in time order.
///
/// # Errors
///
/// Lengths that differ, fewer than two readings, a value or time that is not
/// finite, times that do not increase, a `modulus` or `max_rate` not above 0,
/// a value outside `[0, modulus)`, or a discontinuity outside the readings or
/// sharing a pair with another.
///
/// # Example
///
/// ```
/// use u_analytics::counter::{counter_increments, CounterOptions, IncrementKind};
///
/// // A six-digit register that passes 999 999 between the second and the
/// // third reading.
/// let times = [0.0, 1.0, 2.0];
/// let values = [999_990.0, 999_995.0, 3.0];
/// let options = CounterOptions { modulus: Some(1_000_000.0), ..Default::default() };
/// let steps = counter_increments(&times, &values, &options).unwrap();
/// assert_eq!(steps[1].kind, IncrementKind::Wrap);
/// assert_eq!(steps[1].increase, Some(8.0));
///
/// // Without the modulus the fall is not explained -- and not guessed.
/// let steps = counter_increments(&times, &values, &CounterOptions::default()).unwrap();
/// assert_eq!(steps[1].kind, IncrementKind::Unexplained);
/// assert_eq!(steps[1].increase, None);
/// ```
pub fn counter_increments(
    times: &[f64],
    values: &[f64],
    options: &CounterOptions,
) -> Result<Vec<CounterIncrement>, CounterError> {
    if times.len() != values.len() {
        return Err(CounterError::LengthMismatch {
            times: times.len(),
            values: values.len(),
        });
    }
    if times.len() < 2 {
        return Err(CounterError::TooFewReadings { got: times.len() });
    }
    check_times("times", times)?;
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(CounterError::NotFinite {
                parameter: "values",
                index: Some(i),
            });
        }
    }
    for (parameter, setting) in [("modulus", options.modulus), ("max_rate", options.max_rate)] {
        if let Some(v) = setting {
            if !v.is_finite() {
                return Err(CounterError::NotFinite {
                    parameter,
                    index: None,
                });
            }
            if v <= 0.0 {
                return Err(CounterError::NotPositive { parameter, got: v });
            }
        }
    }
    let within = |parameter: &'static str, index: usize, v: f64| match options.modulus {
        Some(m) if !(0.0..m).contains(&v) => Err(CounterError::OutsideModulus {
            parameter,
            index,
            got: v,
            modulus: m,
        }),
        _ => Ok(()),
    };
    for (i, &v) in values.iter().enumerate() {
        within("values", i, v)?;
    }

    // Each discontinuity to the pair it falls in: after the earlier reading,
    // at or before the later one.
    let mut declared: Vec<Option<&Discontinuity>> = vec![None; times.len() - 1];
    for (i, d) in options.discontinuities.iter().enumerate() {
        for (parameter, v) in [
            ("discontinuities.time", Some(d.time)),
            ("discontinuities.start", Some(d.start)),
            ("discontinuities.last", d.last),
        ] {
            if v.is_some_and(|v| !v.is_finite()) {
                return Err(CounterError::NotFinite {
                    parameter,
                    index: Some(i),
                });
            }
        }
        within("discontinuities.start", i, d.start)?;
        if let Some(last) = d.last {
            within("discontinuities.last", i, last)?;
        }
        let pair = times.partition_point(|&t| t < d.time);
        if pair == 0 || pair == times.len() {
            return Err(CounterError::OutsideReadings {
                index: i,
                got: d.time,
                first: times[0],
                last: times[times.len() - 1],
            });
        }
        let slot = &mut declared[pair - 1];
        if slot.is_some() {
            return Err(CounterError::TwoInOnePair { index: i });
        }
        *slot = Some(d);
    }

    Ok(times
        .windows(2)
        .zip(values.windows(2))
        .zip(declared)
        .map(|((t, v), declared)| {
            let (start, end, before, after) = (t[0], t[1], v[0], v[1]);
            let (increase, kind) = match declared {
                Some(d) => (
                    Some(d.last.map_or(0.0, |last| last - before) + (after - d.start)),
                    IncrementKind::Declared,
                ),
                None if after >= before => (Some(after - before), IncrementKind::Normal),
                None => match options.modulus {
                    Some(m) => {
                        let wrapped = m - before + after;
                        let possible = options
                            .max_rate
                            .is_none_or(|rate| wrapped <= rate * (end - start));
                        if possible {
                            (Some(wrapped), IncrementKind::Wrap)
                        } else {
                            (None, IncrementKind::Unexplained)
                        }
                    }
                    None => (None, IncrementKind::Unexplained),
                },
            };
            CounterIncrement {
                start,
                end,
                increase,
                kind,
            }
        })
        .collect())
}

/// The increments summed over consecutive intervals `[edges[j], edges[j+1])`.
///
/// A pair belongs to the interval its later reading falls in -- its increase
/// is not split across a boundary it straddles (with readings every 15
/// minutes and daily intervals, one pair a day). Pairs ending outside every
/// interval are left out.
///
/// # Errors
///
/// [`CounterError::TooFewReadings`] for fewer than two edges (no interval),
/// [`CounterError::NotFinite`] / [`CounterError::Unordered`] for edges that
/// are not finite or do not increase.
///
/// # Example
///
/// ```
/// use u_analytics::counter::{counter_increments, counter_totals, CounterOptions};
///
/// let times = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0];
/// let values = [100.0, 110.0, 125.0, 130.0, 150.0, 151.0];
/// let steps = counter_increments(&times, &values, &CounterOptions::default()).unwrap();
/// let days = counter_totals(&steps, &[0.0, 24.0, 48.0]).unwrap();
/// assert_eq!(days[0].increase, 30.0); // 6, 12 and 18 end in the first day
/// assert_eq!(days[1].increase, 21.0); // 24 and 30 in the second
/// ```
pub fn counter_totals(
    increments: &[CounterIncrement],
    edges: &[f64],
) -> Result<Vec<CounterTotal>, CounterError> {
    if edges.len() < 2 {
        return Err(CounterError::TooFewReadings { got: edges.len() });
    }
    check_times("edges", edges)?;
    let mut totals: Vec<CounterTotal> = edges
        .windows(2)
        .map(|e| CounterTotal {
            start: e[0],
            end: e[1],
            increase: 0.0,
            pairs: 0,
            unexplained: 0,
        })
        .collect();
    for inc in increments {
        let j = edges.partition_point(|&e| e <= inc.end);
        if j == 0 || j == edges.len() {
            continue;
        }
        let total = &mut totals[j - 1];
        total.pairs += 1;
        match inc.increase {
            Some(x) => total.increase += x,
            None => total.unexplained += 1,
        }
    }
    Ok(totals)
}

fn check_times(parameter: &'static str, times: &[f64]) -> Result<(), CounterError> {
    for (i, t) in times.iter().enumerate() {
        if !t.is_finite() {
            return Err(CounterError::NotFinite {
                parameter,
                index: Some(i),
            });
        }
        if i > 0 && *t <= times[i - 1] {
            return Err(CounterError::Unordered {
                parameter,
                index: i,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(modulus: Option<f64>) -> CounterOptions {
        CounterOptions {
            modulus,
            ..Default::default()
        }
    }

    #[test]
    fn a_rising_counter_has_normal_increments() {
        let s = counter_increments(&[0.0, 1.0, 2.0], &[5.0, 5.0, 9.5], &opts(None)).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(
            (s[0].increase, s[0].kind),
            (Some(0.0), IncrementKind::Normal)
        );
        assert_eq!(
            (s[1].increase, s[1].kind),
            (Some(4.5), IncrementKind::Normal)
        );
        assert_eq!((s[1].start, s[1].end), (1.0, 2.0));
    }

    /// The consumer's day: a six-digit register (999 999.9) passing 0 read
    /// as max − min gave 999 990; the wrap gives the true advance.
    #[test]
    fn a_wrap_is_counted_through_the_modulus() {
        let s =
            counter_increments(&[0.0, 1.0], &[999_995.0, 5.0], &opts(Some(1_000_000.0))).unwrap();
        assert_eq!(s[0].kind, IncrementKind::Wrap);
        assert_eq!(s[0].increase, Some(10.0));
    }

    #[test]
    fn a_fall_without_a_modulus_is_not_guessed() {
        let s = counter_increments(&[0.0, 1.0, 2.0], &[50.0, 10.0, 12.0], &opts(None)).unwrap();
        assert_eq!(
            (s[0].increase, s[0].kind),
            (None, IncrementKind::Unexplained)
        );
        assert_eq!(
            (s[1].increase, s[1].kind),
            (Some(2.0), IncrementKind::Normal)
        );
    }

    /// A reset near the top of the register looks like a wrap; a rate the
    /// counter cannot reach tells them apart.
    #[test]
    fn a_wrap_faster_than_the_counter_can_go_is_unexplained() {
        let o = CounterOptions {
            modulus: Some(1000.0),
            max_rate: Some(10.0),
            ..Default::default()
        };
        // 990 → 5 in 1 time unit: a wrap would be 15 > 10.
        let s = counter_increments(&[0.0, 1.0], &[990.0, 5.0], &o).unwrap();
        assert_eq!(s[0].kind, IncrementKind::Unexplained);
        // 998 → 5: a wrap of 7 is possible.
        let s = counter_increments(&[0.0, 1.0], &[998.0, 5.0], &o).unwrap();
        assert_eq!((s[0].kind, s[0].increase), (IncrementKind::Wrap, Some(7.0)));
    }

    #[test]
    fn a_declared_replacement_counts_from_the_new_start() {
        let mut o = opts(None);
        o.discontinuities = vec![Discontinuity {
            time: 1.5,
            start: 0.0,
            last: None,
        }];
        let s = counter_increments(&[0.0, 1.0, 2.0], &[100.0, 120.0, 7.0], &o).unwrap();
        assert_eq!(
            (s[1].kind, s[1].increase),
            (IncrementKind::Declared, Some(7.0))
        );
        // With the old register's last reading, its last advance counts too.
        o.discontinuities[0].last = Some(126.0);
        let s = counter_increments(&[0.0, 1.0, 2.0], &[100.0, 120.0, 7.0], &o).unwrap();
        assert_eq!(s[1].increase, Some(13.0));
        // A discontinuity at the later reading's time belongs to that pair.
        o.discontinuities[0].time = 2.0;
        let s = counter_increments(&[0.0, 1.0, 2.0], &[100.0, 120.0, 7.0], &o).unwrap();
        assert_eq!(s[1].kind, IncrementKind::Declared);
    }

    #[test]
    fn a_declared_change_takes_precedence_over_a_wrap() {
        let mut o = opts(Some(1000.0));
        o.discontinuities = vec![Discontinuity {
            time: 1.0,
            start: 0.0,
            last: None,
        }];
        let s = counter_increments(&[0.0, 1.0], &[500.0, 3.0], &o).unwrap();
        assert_eq!(
            (s[0].kind, s[0].increase),
            (IncrementKind::Declared, Some(3.0))
        );
    }

    #[test]
    fn refusals_name_what_is_wrong() {
        let e = |t: &[f64], v: &[f64], o: &CounterOptions| counter_increments(t, v, o).unwrap_err();
        assert_eq!(
            e(&[0.0, 1.0], &[1.0], &opts(None)),
            CounterError::LengthMismatch {
                times: 2,
                values: 1
            }
        );
        assert_eq!(
            e(&[0.0], &[1.0], &opts(None)),
            CounterError::TooFewReadings { got: 1 }
        );
        assert_eq!(
            e(&[0.0, 0.0], &[1.0, 2.0], &opts(None)),
            CounterError::Unordered {
                parameter: "times",
                index: 1
            }
        );
        assert_eq!(
            e(&[0.0, 1.0], &[1.0, f64::NAN], &opts(None)),
            CounterError::NotFinite {
                parameter: "values",
                index: Some(1)
            }
        );
        assert_eq!(
            e(&[0.0, 1.0], &[1.0, 2.0], &opts(Some(0.0))),
            CounterError::NotPositive {
                parameter: "modulus",
                got: 0.0
            }
        );
        assert_eq!(
            e(&[0.0, 1.0], &[1.0, 1000.0], &opts(Some(1000.0))),
            CounterError::OutsideModulus {
                parameter: "values",
                index: 1,
                got: 1000.0,
                modulus: 1000.0
            }
        );
        let mut o = opts(None);
        o.discontinuities = vec![Discontinuity {
            time: 0.0,
            start: 0.0,
            last: None,
        }];
        assert_eq!(
            e(&[0.0, 1.0], &[1.0, 2.0], &o),
            CounterError::OutsideReadings {
                index: 0,
                got: 0.0,
                first: 0.0,
                last: 1.0
            }
        );
        o.discontinuities = vec![
            Discontinuity {
                time: 0.5,
                start: 0.0,
                last: None,
            },
            Discontinuity {
                time: 0.7,
                start: 0.0,
                last: None,
            },
        ];
        assert_eq!(
            e(&[0.0, 1.0], &[1.0, 2.0], &o),
            CounterError::TwoInOnePair { index: 1 }
        );
    }

    #[test]
    fn totals_attribute_each_pair_to_the_interval_its_end_falls_in() {
        let s = counter_increments(
            &[0.0, 10.0, 20.0, 30.0, 40.0],
            &[0.0, 5.0, 2.0, 4.0, 9.0],
            &opts(None),
        )
        .unwrap();
        let t = counter_totals(&s, &[0.0, 20.0, 40.0, 60.0]).unwrap();
        // [0, 20): the pair ending at 10. [20, 40): ending at 20 (fall,
        // unexplained) and 30. [40, 60): ending at 40.
        assert_eq!((t[0].increase, t[0].pairs, t[0].unexplained), (5.0, 1, 0));
        assert_eq!((t[1].increase, t[1].pairs, t[1].unexplained), (2.0, 2, 1));
        assert_eq!((t[2].increase, t[2].pairs, t[2].unexplained), (5.0, 1, 0));
        assert!(counter_totals(&s, &[0.0]).is_err());
        assert!(counter_totals(&s, &[10.0, 5.0]).is_err());
    }
}

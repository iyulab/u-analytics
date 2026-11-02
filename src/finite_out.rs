//! Test-only guard: a response must not carry a NaN or an infinity in a
//! number field.
//!
//! JSON has no non-finite numbers, so `serde_json` writes one as `null` — a
//! value the result declares a `number` arrives as `null`, and a typed client
//! (a `.d.ts`, a C# record) fails to read it or reads it wrong. A value with
//! no meaning in some case is declared `Option` and sent as `None`.
//! [`assert_finite`] walks a response the way a serializer would and fails at
//! the first non-finite number, naming its path; the bindings run it on every
//! response they write in a test build, so every test that reaches a response
//! checks it.

use serde::ser::{self, Serialize};
use std::fmt;

#[derive(Debug)]
pub(crate) struct NonFinite(String);

impl fmt::Display for NonFinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NonFinite {}

impl ser::Error for NonFinite {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        NonFinite(msg.to_string())
    }
}

/// Panics when `value` serializes a NaN or an infinity, naming where.
pub(crate) fn assert_finite<T: Serialize + ?Sized>(value: &T) {
    if let Err(e) = value.serialize(Walk {
        path: String::new(),
    }) {
        panic!("a response carries a non-finite number, which JSON writes as null: {e}");
    }
}

struct Walk {
    path: String,
}

impl Walk {
    fn at(&self, key: impl fmt::Display) -> Walk {
        Walk {
            path: format!("{}/{key}", self.path),
        }
    }
    fn number(&self, v: f64) -> Result<(), NonFinite> {
        if v.is_finite() {
            Ok(())
        } else {
            Err(NonFinite(format!("{v} at `{}`", self.path)))
        }
    }
}

struct Seq {
    walk: Walk,
    index: usize,
}

impl Seq {
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), NonFinite> {
        let r = value.serialize(self.walk.at(self.index));
        self.index += 1;
        r
    }
}

impl ser::SerializeSeq for Seq {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> {
        self.element(v)
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}
impl ser::SerializeTuple for Seq {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> {
        self.element(v)
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}
impl ser::SerializeTupleStruct for Seq {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> {
        self.element(v)
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}
impl ser::SerializeTupleVariant for Seq {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> {
        self.element(v)
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}

struct Map {
    walk: Walk,
    key: String,
}

impl ser::SerializeMap for Map {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), NonFinite> {
        self.key = serde_json::to_string(key).unwrap_or_default();
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> {
        v.serialize(self.walk.at(&self.key))
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}
impl ser::SerializeStruct for Map {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        v: &T,
    ) -> Result<(), NonFinite> {
        v.serialize(self.walk.at(key))
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}
impl ser::SerializeStructVariant for Map {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        v: &T,
    ) -> Result<(), NonFinite> {
        v.serialize(self.walk.at(key))
    }
    fn end(self) -> Result<(), NonFinite> {
        Ok(())
    }
}

impl ser::Serializer for Walk {
    type Ok = ();
    type Error = NonFinite;
    type SerializeSeq = Seq;
    type SerializeTuple = Seq;
    type SerializeTupleStruct = Seq;
    type SerializeTupleVariant = Seq;
    type SerializeMap = Map;
    type SerializeStruct = Map;
    type SerializeStructVariant = Map;

    fn serialize_f64(self, v: f64) -> Result<(), NonFinite> {
        self.number(v)
    }
    fn serialize_f32(self, v: f32) -> Result<(), NonFinite> {
        self.number(f64::from(v))
    }
    fn serialize_bool(self, _: bool) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_i8(self, _: i8) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_i16(self, _: i16) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_i32(self, _: i32) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_i64(self, _: i64) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_i128(self, _: i128) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_u8(self, _: u8) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_u16(self, _: u16) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_u32(self, _: u32) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_u64(self, _: u64) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_u128(self, _: u128) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_char(self, _: char) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_str(self, _: &str) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_none(self) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), NonFinite> {
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<(), NonFinite> {
        v.serialize(self.at(variant))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Seq, NonFinite> {
        Ok(Seq {
            walk: self,
            index: 0,
        })
    }
    fn serialize_tuple(self, _: usize) -> Result<Seq, NonFinite> {
        Ok(Seq {
            walk: self,
            index: 0,
        })
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Seq, NonFinite> {
        Ok(Seq {
            walk: self,
            index: 0,
        })
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Seq, NonFinite> {
        Ok(Seq {
            walk: self.at(variant),
            index: 0,
        })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Map, NonFinite> {
        Ok(Map {
            walk: self,
            key: String::new(),
        })
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Map, NonFinite> {
        Ok(Map {
            walk: self,
            key: String::new(),
        })
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Map, NonFinite> {
        Ok(Map {
            walk: self.at(variant),
            key: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct R {
        a: f64,
        b: Option<f64>,
        c: Vec<f64>,
    }

    #[test]
    fn a_finite_response_passes_and_a_null_option_is_not_a_number() {
        assert_finite(&R {
            a: 1.0,
            b: None,
            c: vec![2.0],
        });
    }

    #[test]
    #[should_panic(expected = "at `/c/1`")]
    fn an_infinity_is_named_by_its_path() {
        assert_finite(&R {
            a: 1.0,
            b: Some(2.0),
            c: vec![2.0, f64::INFINITY],
        });
    }
}

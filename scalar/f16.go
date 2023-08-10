package scalar

import (
	"fmt"
	"math"
)

type F16 struct {
	Value uint16
}

// Constructors
func F16fromF32(value float32) *F16 {
	return &F16{Value: f32bitsToF16bits(math.Float32bits(value))}
}

func F16fromF64(value float64) *F16 {
	return F16fromF32(float32(value))
}

// Getter and setter
func (p *F16) G32() float32 {
	u32 := f16bitsToF32bits(p.Value)
	return math.Float32frombits(u32)
}

func (p *F16) G64() float64 {
	return float64(p.G32())
}

func (p *F16) S32(v float32) {
	p.Value = f32bitsToF16bits(math.Float32bits(v))
}

func (p *F16) S64(v float64) {
	p.S32(float32(v))
}

func (p *F16) Copy() Scalar {
	return &F16{p.Value}
}

// Math
func (p *F16) Add(v Scalar) Scalar {
	return F16fromF32(p.G32() + v.G32())
}

func (p *F16) Sub(v Scalar) Scalar {
	return F16fromF32(p.G32() - v.G32())
}

func (p *F16) Mul(v Scalar) Scalar {
	return F16fromF32(p.G32() * v.G32())
}

func (p *F16) Div(v Scalar) Scalar {
	return F16fromF32(p.G32() / v.G32())
}

// Self Math
func (p *F16) SelfAdd(v Scalar) {
	p.S32(p.G32() + v.G32())
}

func (p *F16) SelfSub(v Scalar) {
	p.S32(p.G32() - v.G32())
}

func (p *F16) SelfMul(v Scalar) {
	p.S32(p.G32() * v.G32())
}

func (p *F16) SelfDiv(v Scalar) {
	p.S32(p.G32() / v.G32())
}

func (p *F16) Cmp(v Scalar) CmpRes {
	if p.G32() < v.G32() {
		return CmpL
	}
	if p.G32() > v.G32() {
		return CmpG
	}
	return CmpE
}

// f16bitsToF32bits returns uint32 (float32 bits) converted from specified uint16.
func f16bitsToF32bits(in uint16) uint32 {
	// All 65536 conversions with this were confirmed to be correct
	// by Montgomery Edwards⁴⁴⁸ (github.com/x448).

	sign := uint32(in&0x8000) << 16 // sign for 32-bit
	exp := uint32(in&0x7c00) >> 10  // exponenent for 16-bit
	coef := uint32(in&0x03ff) << 13 // significand for 32-bit

	if exp == 0x1f {
		if coef == 0 {
			// infinity
			return sign | 0x7f800000 | coef
		}
		// NaN
		return sign | 0x7fc00000 | coef
	}

	if exp == 0 {
		if coef == 0 {
			// zero
			return sign
		}

		// normalize subnormal numbers
		exp++
		for coef&0x7f800000 == 0 {
			coef <<= 1
			exp--
		}
		coef &= 0x007fffff
	}

	return sign | ((exp + (0x7f - 0xf)) << 23) | coef
}

// f32bitsToF16bits returns uint16 (Float16 bits) converted from the specified float32.
// Conversion rounds to nearest integer with ties to even.
func f32bitsToF16bits(u32 uint32) uint16 {
	// Translated from Rust to Go by Montgomery Edwards⁴⁴⁸ (github.com/x448).
	// All 4294967296 conversions with this were confirmed to be correct by x448.
	// Original Rust implementation is by Kathryn Long (github.com/starkat99) with MIT license.

	sign := u32 & 0x80000000
	exp := u32 & 0x7f800000
	coef := u32 & 0x007fffff

	if exp == 0x7f800000 {
		// NaN or Infinity
		nanBit := uint32(0)
		if coef != 0 {
			nanBit = uint32(0x0200)
		}
		return uint16((sign >> 16) | uint32(0x7c00) | nanBit | (coef >> 13))
	}

	halfSign := sign >> 16

	unbiasedExp := int32(exp>>23) - 127
	halfExp := unbiasedExp + 15

	if halfExp >= 0x1f {
		return uint16(halfSign | uint32(0x7c00))
	}

	if halfExp <= 0 {
		if 14-halfExp > 24 {
			return uint16(halfSign)
		}
		c := coef | uint32(0x00800000)
		halfCoef := c >> uint32(14-halfExp)
		roundBit := uint32(1) << uint32(13-halfExp)
		if (c&roundBit) != 0 && (c&(3*roundBit-1)) != 0 {
			halfCoef++
		}
		return uint16(halfSign | halfCoef)
	}

	uHalfExp := uint32(halfExp) << 10
	halfCoef := coef >> 13
	roundBit := uint32(0x00001000)
	if (coef&roundBit) != 0 && (coef&(3*roundBit-1)) != 0 {
		return uint16((halfSign | uHalfExp | halfCoef) + 1)
	}
	return uint16(halfSign | uHalfExp | halfCoef)
}

func (p *F16) IsNaN() bool {
	return (p.Value&0x7c00 == 0x7c00) && (p.Value&0x03ff != 0)
}

func (p *F16) IsQuietNaN() bool {
	return (p.Value&0x7c00 == 0x7c00) && (p.Value&0x03ff != 0) && (p.Value&0x0200 != 0)
}

func (p *F16) IsInf(sign int) bool {
	return ((p.Value == 0x7c00) && sign >= 0) ||
		(p.Value == 0xfc00 && sign <= 0)
}

func (p *F16) IsFinite() bool {
	return (p.Value & 0x7c00) != 0x7c00
}

func (p *F16) IsNormal() bool {
	exp := p.Value & 0x7c00
	return (exp != 0x7c00) && (exp != 0)
}

func (p *F16) Signbit() bool {
	return (p.Value & 0x8000) != 0
}

func F16NaN() F16 {
	return F16{0x7e01}
}

func F16Inf(sign int) F16 {
	if sign >= 0 {
		return F16{0x7c00}
	}
	return F16{0x8000 | 0x7c00}
}

func (p *F16) String() string {
	return fmt.Sprint(p.G32())
}

package scalar

import "fmt"

type F32 struct {
	Value float32
}

// Constructors
func NewF32(value float32) *F32 {
	return &F32{
		Value: value,
	}
}

func F32fromF64(value float64) *F32 {
	return &F32{
		Value: float32(value),
	}
}

// Getter and setters
func (p *F32) G32() float32      { return p.Value }
func (p *F32) G64() float64      { return float64(p.Value) }
func (p *F32) S32(value float32) { p.Value = value }
func (p *F32) S64(value float64) { p.Value = float32(value) }

func (p *F32) Copy() Scalar { return &F32{Value: p.Value} }

// Math
func (p *F32) Add(v Scalar) Scalar { return &F32{Value: p.Value + v.G32()} }
func (p *F32) Sub(v Scalar) Scalar { return &F32{Value: p.Value - v.G32()} }
func (p *F32) Mul(v Scalar) Scalar { return &F32{Value: p.Value * v.G32()} }
func (p *F32) Div(v Scalar) Scalar { return &F32{Value: p.Value / v.G32()} }

// Self Math
func (p *F32) SelfAdd(v Scalar) { p.S32(p.Value + v.G32()) }
func (p *F32) SelfSub(v Scalar) { p.S32(p.Value - v.G32()) }
func (p *F32) SelfMul(v Scalar) { p.S32(p.Value * v.G32()) }
func (p *F32) SelfDiv(v Scalar) { p.S32(p.Value / v.G32()) }

func (p *F32) Cmp(v Scalar) CmpRes {
	if p.G32() < v.G32() {
		return CmpL
	}
	if p.G32() > v.G32() {
		return CmpG
	}
	return CmpE
}

func (p *F32) String() string {
	return fmt.Sprint(p.G32())
}

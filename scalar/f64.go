package scalar

import "fmt"

type F64 struct {
	Value float64
}

// Constructors
func NewF64(value float64) *F64 {
	return &F64{
		Value: value,
	}
}

func F64fromF32(value float32) *F64 {
	return &F64{
		Value: float64(value),
	}
}

// Getter and setters
func (p *F64) G32() float32      { return float32(p.Value) }
func (p *F64) G64() float64      { return p.Value }
func (p *F64) S32(value float32) { p.Value = float64(value) }
func (p *F64) S64(value float64) { p.Value = value }

func (p *F64) Copy() Scalar { return &F64{Value: p.Value} }

// Math
func (p *F64) Add(v Scalar) Scalar { return &F64{Value: p.Value + v.G64()} }
func (p *F64) Sub(v Scalar) Scalar { return &F64{Value: p.Value - v.G64()} }
func (p *F64) Mul(v Scalar) Scalar { return &F64{Value: p.Value * v.G64()} }
func (p *F64) Div(v Scalar) Scalar { return &F64{Value: p.Value / v.G64()} }

// Self Math
func (p *F64) SelfAdd(v Scalar) { p.S64(p.Value + v.G64()) }
func (p *F64) SelfSub(v Scalar) { p.S64(p.Value - v.G64()) }
func (p *F64) SelfMul(v Scalar) { p.S64(p.Value * v.G64()) }
func (p *F64) SelfDiv(v Scalar) { p.S64(p.Value / v.G64()) }

func (p *F64) Cmp(v Scalar) CmpRes {
	if p.G64() < v.G64() {
		return CmpL
	}
	if p.G64() > v.G64() {
		return CmpG
	}
	return CmpE
}

func (p *F64) String() string {
	return fmt.Sprint(p.G32())
}

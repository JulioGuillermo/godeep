package scalar

type CmpRes byte

const (
	CmpL = CmpRes(iota)
	CmpE
	CmpG
)

type Scalar interface {
	G32() float32
	G64() float64
	S32(float32)
	S64(float64)

	Add(v Scalar) Scalar
	Sub(v Scalar) Scalar
	Mul(v Scalar) Scalar
	Div(v Scalar) Scalar

	SelfAdd(v Scalar)
	SelfSub(v Scalar)
	SelfMul(v Scalar)
	SelfDiv(v Scalar)

	Cmp(v Scalar) CmpRes

	Copy() Scalar
	String() string
}

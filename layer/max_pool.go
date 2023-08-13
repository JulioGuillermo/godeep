package layer

import (
	"fmt"

	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type MaxPool2D[T types.Number] struct {
	Base[T]
	SX uint
	SY uint
}

func NewMaxPool2D[T types.Number](s uint) Layer[T] {
	l := &MaxPool2D[T]{}
	l.Type = "MaxPool"
	l.SX = s
	l.SY = s

	return l
}

func NewMaxPool2Dxy[T types.Number](sx, sy uint) Layer[T] {
	l := &MaxPool2D[T]{}
	l.Type = "MaxPool"
	l.SX = sx
	l.SY = sy

	return l
}

func (p *MaxPool2D[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	err := p.PreBuild()
	if err != nil {
		return 0, err
	}

	if p.PreLayer == nil {
		return 0, p.Error("This layer can not be input layer")
	}
	p.Activation = p.PreLayer.GetActivation()
	return p.Index, nil
}

func (p *MaxPool2D[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}
	if len(p.Input.GetShape()) != 3 {
		return p.Error(fmt.Sprintf(
			"Invalid input shape %s, you can use a reshape layer",
			tools.ShapeStr(p.Input.GetShape()),
		))
	}

	inShape := p.Input.GetShape()
	outShape := p.Input.GetShape()
	outShape[1] = (outShape[1] + 1) / p.SX
	outShape[2] = (outShape[2] + 1) / p.SY
	p.Neta = tensor.NewZeros[T](outShape...)
	p.Output = p.Neta // tensor.NewZeros[T](outShape...)

	for f := uint(0); f < outShape[0]; f++ {
		for x := uint(0); x < outShape[1]; x++ {
			for y := uint(0); y < outShape[2]; y++ {
				n, err := p.Neta.GetOperand(f, x, y)
				if err != nil {
					return err
				}
				ops := make([]*number.Scalar[T], 0, p.SX*p.SY)
				for i := uint(0); i < p.SX; i++ {
					ix := x*p.SX + i
					if ix >= inShape[1] {
						continue
					}
					for j := uint(0); j < p.SY; j++ {
						iy := y*p.SY + j
						if iy >= inShape[2] {
							continue
						}
						in, err := p.Input.GetOperand(f, ix, iy)
						if err != nil {
							return err
						}
						ops = append(ops, in)
					}
				}
				ctx.Push(&operation.Max[T]{
					Scalar: n,
					Args:   ops,
				})
			}
		}
	}

	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)

	return nil
}

func (p *MaxPool2D[T]) BuildBackpropagation(ctx *context.Context, a, m *number.Scalar[T]) error {
	if p.CheckBP() {
		return nil
	}

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}

	inShape := p.Input.GetShape()
	outShape := p.Output.GetShape()

	for f := uint(0); f < outShape[0]; f++ {
		for x := uint(0); x < outShape[1]; x++ {
			for y := uint(0); y < outShape[2]; y++ {
				dif, err := Dif.GetOperand(f, x, y)
				if err != nil {
					return err
				}
				n, err := p.Neta.GetOperand(f, x, y)
				if err != nil {
					return err
				}
				for i := uint(0); i < p.SX; i++ {
					ix := x*p.SX + i
					if ix >= inShape[1] {
						continue
					}
					for j := uint(0); j < p.SY; j++ {
						iy := y*p.SY + j
						if iy >= inShape[2] {
							continue
						}
						pdif, err := p.PreLayer.GetDif().GetOperand(f, ix, iy)
						if err != nil {
							return err
						}
						in, err := p.Input.GetOperand(f, ix, iy)
						if err != nil {
							return err
						}

						eq := &number.Scalar[T]{}
						ctx.Push(&operation.Equals[T]{
							Scalar: eq,
							A:      in,
							B:      n,
						})
						d := &number.Scalar[T]{}
						ctx.Push(&operation.Mul[T]{
							Scalar: d,
							A:      dif,
							B:      eq,
						})
						ctx.Push(&operation.Set[T]{
							Scalar: pdif,
							O:      d,
						})
					}
				}
			}
		}
	}

	return p.PostBuildBackpropagation(ctx, a, m)
}

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

type UpSampling2D[T types.Number] struct {
	Base[T]
	SX uint
	SY uint
}

func NewUpSampling2D[T types.Number](s uint) Layer[T] {
	l := &UpSampling2D[T]{}
	l.Type = "MaxPool"
	l.SX = s
	l.SY = s

	return l
}

func NewUpSampling2Dxy[T types.Number](sx, sy uint) Layer[T] {
	l := &MaxPool2D[T]{}
	l.Type = "MaxPool"
	l.SX = sx
	l.SY = sy

	return l
}

func (p *UpSampling2D[T]) Build() (uint, error) {
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
	// p.Activation = p.PreLayer.GetActivation()
	return p.Index, nil
}

func (p *UpSampling2D[T]) BuildFeedforward(ctx *context.Context) error {
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

	// inShape := p.Input.GetShape()
	outShape := p.Input.GetShape()
	outShape[1] = outShape[1] * p.SX
	outShape[2] = outShape[2] * p.SY
	p.Output = tensor.NewZeros[T](outShape...)
	// p.Output = p.Neta // tensor.NewZeros[T](outShape...)

	for f := uint(0); f < outShape[0]; f++ {
		for x := uint(0); x < outShape[1]; x++ {
			for y := uint(0); y < outShape[2]; y++ {
				o, err := p.Output.GetOperand(f, x, y)
				if err != nil {
					return err
				}
				inX := x / p.SX
				inY := y / p.SY

				i, err := p.Input.GetOperand(f, inX, inY)
				if err != nil {
					return err
				}

				ctx.Push(&operation.Set[T]{
					Scalar: o,
					O:      i,
				})
			}
		}
	}

	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)

	return nil
}

func (p *UpSampling2D[T]) BuildBackpropagation(ctx *context.Context, a, m *number.Scalar[T]) error {
	if p.CheckBP() {
		return nil
	}

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}

	err := Dif.BuildGraph(ctx)
	if err != nil {
		return err
	}

	inShape := p.Input.GetShape()

	preDif := p.PreLayer.GetDif()

	for f := uint(0); f < inShape[0]; f++ {
		for x := uint(0); x < inShape[1]; x++ {
			for y := uint(0); y < inShape[2]; y++ {
				pdif, err := preDif.GetOperand(f, x, y)
				if err != nil {
					return err
				}
				ops := make([]*number.Scalar[T], 0, p.SX*p.SY)

				for i := uint(0); i < p.SX; i++ {
					oX := x*p.SX + i
					for j := uint(0); j < p.SY; j++ {
						oY := y*p.SY + j
						dif, err := Dif.GetOperand(f, oX, oY)
						if err != nil {
							return err
						}
						ops = append(ops, dif)
					}
				}

				ctx.Push(&operation.Sum[T]{
					Scalar: pdif,
					Args:   ops,
				})
			}
		}
	}

	return p.PostBuildBackpropagation(ctx, a, m)
}

func (p *UpSampling2D[T]) BuildDer(ctx *context.Context) (tensor.Tensor[T], error) {
	if p.Der == nil {
		outShape := p.Output.GetShape()

		p.Der = tensor.NewZeros[T](outShape...)

		preDer, err := p.PreLayer.BuildDer(ctx)
		if err != nil {
			return nil, err
		}

		for f := uint(0); f < outShape[0]; f++ {
			for x := uint(0); x < outShape[1]; x++ {
				for y := uint(0); y < outShape[2]; y++ {
					o, err := p.Der.GetOperand(f, x, y)
					if err != nil {
						return nil, err
					}
					inX := x / p.SX
					inY := y / p.SY

					i, err := preDer.GetOperand(f, inX, inY)
					if err != nil {
						return nil, err
					}

					ctx.Push(&operation.Set[T]{
						Scalar: o,
						O:      i,
					})
				}
			}
		}
	}
	return p.Der, nil
}

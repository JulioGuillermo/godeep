package layer

import (
	"fmt"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type Conv2D[T types.Number] struct {
	Base[T]
	NF uint
	KW uint
	KH uint
	SX uint
	SY uint
}

func NewConv2D[T types.Number](
	filters, kernelSize, strideSize uint,
	act activation.Activation[T],
) Layer[T] {
	l := &Conv2D[T]{}
	l.Type = "Conv2D"
	l.Activation = act
	l.NF = filters
	l.KW = kernelSize
	l.KH = kernelSize
	l.SX = strideSize
	l.SY = strideSize
	l.Trainable = true

	return l
}

func NewConv2Dwh[T types.Number](
	filters, kernelWidth, kernelHeight, strideWidth, strideHeight uint,
	act activation.Activation[T],
) Layer[T] {
	l := &Conv2D[T]{}
	l.Type = "Conv2D"
	l.Activation = act
	l.NF = filters
	l.KW = kernelWidth
	l.KH = kernelHeight
	l.SX = strideWidth
	l.SY = strideHeight
	l.Trainable = true

	return l
}

func (p *Conv2D[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	return p.Index, p.PreBuild()
}

func (p *Conv2D[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}

	if p.Input == nil {
		return p.Error("Not input defined, please use an input layer first")
	}
	if len(p.Input.GetShape()) != 3 {
		return p.Error(fmt.Sprintf(
			"Invalid input shape %s, you can use a reshape layer",
			tools.ShapeStr(p.Input.GetShape()),
		))
	}

	inShape := p.Input.GetShape()
	outShape := []uint{
		p.NF,
		(inShape[1]-p.KW)/p.SX + 1,
		(inShape[2]-p.KH)/p.SY + 1,
	}

	p.Neta = tensor.NewZeros[T](outShape...)
	p.Output = tensor.NewZeros[T](outShape...)
	p.Dif = tensor.NewZeros[T](outShape...)
	p.Bias = tensor.NewNormRand[T](outShape...)
	p.Weights = tensor.NewNormRand[T](outShape[0], inShape[0], p.KW, p.KH)

	err = tensor.Transfer(ctx, p.Bias, p.Neta)
	if err != nil {
		return err
	}

	for f := uint(0); f < outShape[0]; f++ {
		for x := uint(0); x < outShape[1]; x++ {
			offsetx := x * p.SX
			row := tensor.SubExtendedTensor(p.Input, 1, offsetx, offsetx+p.KW)
			for y := uint(0); y < outShape[2]; y++ {
				offsety := y * p.SY
				cell := tensor.SubExtendedTensor(row, 2, offsety, offsety+p.KH)
				weights := tensor.SubTensor(p.Weights, 0, f, f+1)
				weights = tensor.Reshape(weights, inShape[0], p.KW, p.KH)
				CxW := tensor.Mul(cell, weights)
				sum := tensor.Sum(CxW)
				err := sum.BuildGraph(ctx)
				if err != nil {
					return err
				}
				neta, err := p.Neta.GetOperand(f, x, y)
				if err != nil {
					return err
				}
				ctx.Push(&operation.Add[T]{
					Scalar: neta,
					A:      neta,
					B:      sum.GetOperands()[0],
				})
			}
		}
	}

	p.Output = tensor.Activate(p.Neta, p.Activation.Activate)
	return p.Output.BuildGraph(ctx)
}

func (p *Conv2D[T]) BuildBackpropagation(
	ctx *context.Context,
	Alpha, Momentum *number.Scalar[T],
) error {
	if p.CheckBP() {
		return nil
	}

	inShape := p.Input.GetShape()
	outShape := p.Output.GetShape()

	p.NWeights = tensor.NewZeros[T](p.Weights.GetShape()...)
	p.MWeights = tensor.NewZeros[T](p.Weights.GetShape()...)

	// p.NBias = tensor.NewZeros[T](p.Bias.GetShape()...)
	p.MBias = tensor.NewZeros[T](p.Bias.GetShape()...)

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}

	if p.PreLayer != nil {
		der := tensor.Activate(p.PreLayer.GetNetas(), p.PreLayer.GetActivation().Derive)
		err := der.BuildGraph(ctx)
		if err != nil {
			return err
		}
		preDif := p.PreLayer.GetDif()

		for f := uint(0); f < outShape[0]; f++ {
			for x := uint(0); x < outShape[1]; x++ {
				for y := uint(0); y < outShape[2]; y++ {
					for i := uint(0); i < inShape[0]; i++ {
						for kx := uint(0); kx < p.KW; kx++ {
							for ky := uint(0); ky < p.KW; ky++ {
								ix := x*p.SX + kx
								iy := y*p.SY + ky
								if ix >= inShape[1] || iy >= inShape[2] {
									continue
								}
								w, err := p.Weights.GetOperand(f, i, kx, ky)
								if err != nil {
									return err
								}
								dw := &number.Scalar[T]{}
								d, err := der.GetOperand(i, ix, iy)
								if err != nil {
									return err
								}
								ctx.Push(&operation.Mul[T]{
									Scalar: dw,
									A:      d,
									B:      w,
								})
								pd, err := preDif.GetOperand(i, ix, iy)
								if err != nil {
									return err
								}
								ctx.Push(&operation.Add[T]{
									Scalar: pd,
									A:      pd,
									B:      dw,
								})
							}
						}
					}
				}
			}
		}
	}

	dif := tensor.MulScalar(p.Dif, Alpha)
	biasMom := tensor.MulScalar(p.MBias, Momentum)
	deltaBias := tensor.Add(dif, biasMom)

	p.NBias = tensor.Add(p.Bias, deltaBias)
	err := p.NBias.BuildGraph(ctx)
	if err != nil {
		return err
	}

	err = tensor.Transfer(ctx, deltaBias, p.MBias)
	if err != nil {
		return err
	}

	weightsMom := tensor.MulScalar(p.MWeights, Momentum)
	err = weightsMom.BuildGraph(ctx)
	if err != nil {
		return err
	}

	var in *number.Scalar[T]
	for f := uint(0); f < outShape[0]; f++ {
		for i := uint(0); i < inShape[0]; i++ {
			for kx := uint(0); kx < p.KW; kx++ {
				for ky := uint(0); ky < p.KH; ky++ {
					w, err := p.Weights.GetOperand(f, i, kx, ky)
					if err != nil {
						return err
					}
					m, err := weightsMom.GetOperand(f, i, kx, ky)
					if err != nil {
						return err
					}
					nw, err := p.NWeights.GetOperand(f, i, kx, ky)
					if err != nil {
						return err
					}
					ops := make([]*number.Scalar[T], 0, outShape[1]*outShape[2])
					for ox := uint(0); ox < outShape[1]; ox++ {
						for oy := uint(0); oy < outShape[2]; oy++ {
							ix := ox*p.SY + kx
							iy := oy*p.SY + ky
							if ix < inShape[1] || iy < inShape[2] {
								in, err = p.Input.GetOperand(i, ix, iy)
								if err != nil {
									return err
								}
							} else {
								in = &number.Scalar[T]{}
							}
							dif, err := dif.GetOperand(f, ox, oy)
							if err != nil {
								return err
							}
							inDif := &number.Scalar[T]{}
							ctx.Push(&operation.Mul[T]{
								Scalar: inDif,
								A:      dif,
								B:      in,
							})
							ops = append(ops, inDif)
						}
					}
					dw := &number.Scalar[T]{}
					ctx.Push(&operation.Sum[T]{
						Scalar: dw,
						Args:   ops,
					})
					ctx.Push(&operation.Sum[T]{
						Scalar: nw,
						Args:   []*number.Scalar[T]{w, dw, m},
					})
					ctx.Push(&operation.Set[T]{
						Scalar: m,
						O:      dw,
					})
				}
			}
		}
	}

	return p.PostBuildBackpropagation(ctx, Alpha, Momentum)
}

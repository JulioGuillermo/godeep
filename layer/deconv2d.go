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

type Deconv2D[T types.Number] struct {
	Base[T]
	NF uint
	KW uint
	KH uint
	SX uint
	SY uint
}

func NewDeconv2D[T types.Number](
	filters, kernelSize, strideSize uint,
	act activation.Activation[T],
) Layer[T] {
	l := &Deconv2D[T]{}
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

func NewDeconv2Dwh[T types.Number](
	filters, kernelWidth, kernelHeight, strideWidth, strideHeight uint,
	act activation.Activation[T],
) Layer[T] {
	l := &Deconv2D[T]{}
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

func (p *Deconv2D[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	return p.Index, p.PreBuild()
}

func (p *Deconv2D[T]) BuildFeedforward(ctx *context.Context) error {
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
		(inShape[1]-1)*p.SX + p.KW,
		(inShape[2]-1)*p.SY + p.KH,
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

	err = tensor.Transfer(ctx, p.Bias, p.Neta)
	if err != nil {
		return err
	}

	for outF := uint(0); outF < outShape[0]; outF++ {
		for inF := uint(0); inF < inShape[0]; inF++ {
			for x := uint(0); x < inShape[1]; x++ {
				offsetx := x * p.SX
				for y := uint(0); y < inShape[2]; y++ {
					offsety := y * p.SY

					for kx := uint(0); kx < p.KW; kx++ {
						outX := offsetx + kx
						for ky := uint(0); ky < p.KH; ky++ {
							outY := offsety + ky

							out, err := p.Neta.GetOperand(outF, outX, outY)
							if err != nil {
								return err
							}

							in, err := p.Input.GetOperand(inF, x, y)
							if err != nil {
								return err
							}

							w, err := p.Weights.GetOperand(outF, inF, kx, ky)
							if err != nil {
								return err
							}

							inW := &number.Scalar[T]{}
							ctx.Push(&operation.Mul[T]{
								Scalar: inW,
								A:      in,
								B:      w,
							})

							ctx.Push(&operation.Add[T]{
								Scalar: out,
								A:      out,
								B:      inW,
							})
						}
					}
				}
			}
		}
	}

	p.Output = tensor.Activate(p.Neta, p.Activation.Activate)
	return p.Output.BuildGraph(ctx)
}

func (p *Deconv2D[T]) BuildBackpropagation(
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
	err := Dif.BuildGraph(ctx)
	if err != nil {
		return err
	}

	if p.PreLayer != nil {
		der, err := p.PreLayer.BuildDer(ctx)
		if err != nil {
			return err
		}
		preDif := p.PreLayer.GetDif()

		for inF := uint(0); inF < inShape[0]; inF++ {
			for x := uint(0); x < inShape[1]; x++ {
				offsetx := x * p.SX
				for y := uint(0); y < inShape[2]; y++ {
					offsety := y * p.SY

					pder, err := der.GetOperand(inF, x, y)
					if err != nil {
						return err
					}
					pdif, err := preDif.GetOperand(inF, x, y)
					if err != nil {
						return err
					}
					ops := make([]*number.Scalar[T], 0, outShape[0]*p.KW*p.KH)

					for outF := uint(0); outF < outShape[0]; outF++ {
						for kx := uint(0); kx < p.KW; kx++ {
							outX := offsetx + kx
							for ky := uint(0); ky < p.KH; ky++ {
								outY := offsety + ky

								dif, err := Dif.GetOperand(outF, outX, outY)
								if err != nil {
									return err
								}
								w, err := p.Weights.GetOperand(outF, inF, kx, ky)
								if err != nil {
									return err
								}

								difW := &number.Scalar[T]{}
								ctx.Push(&operation.Mul[T]{
									Scalar: difW,
									A:      dif,
									B:      w,
								})

								ops = append(ops, difW)
							}
						}
					}

					ndif := &number.Scalar[T]{}
					ctx.Push(&operation.Sum[T]{
						Scalar: ndif,
						Args:   ops,
					})

					ctx.Push(&operation.Mul[T]{
						Scalar: ndif,
						A:      ndif,
						B:      pder,
					})

					ctx.Push(&operation.Add[T]{
						Scalar: pdif,
						A:      pdif,
						B:      ndif,
					})
				}
			}
		}
	}

	dif := tensor.MulScalar(p.Dif, Alpha)
	biasMom := tensor.MulScalar(p.MBias, Momentum)
	deltaBias := tensor.Add(dif, biasMom)

	p.NBias = tensor.Add(p.Bias, deltaBias)
	err = p.NBias.BuildGraph(ctx)
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

	for outF := uint(0); outF < outShape[0]; outF++ {
		for kx := uint(0); kx < p.KW; kx++ {
			for ky := uint(0); ky < p.KH; ky++ {
				for inF := uint(0); inF < inShape[0]; inF++ {
					w, err := p.Weights.GetOperand(outF, inF, kx, ky)
					if err != nil {
						return err
					}
					mw, err := weightsMom.GetOperand(outF, inF, kx, ky)
					if err != nil {
						return err
					}
					nw, err := p.NWeights.GetOperand(outF, inF, kx, ky)
					if err != nil {
						return err
					}

					ops := make([]*number.Scalar[T], 0, inShape[1]*inShape[2])
					for x := uint(0); x < inShape[1]; x++ {
						outX := x*p.SX + kx
						for y := uint(0); y < inShape[2]; y++ {
							outY := y*p.SY + ky

							in, err := p.Input.GetOperand(inF, x, y)
							if err != nil {
								return err
							}

							d, err := dif.GetOperand(outF, outX, outY)
							if err != nil {
								return err
							}

							din := &number.Scalar[T]{}
							ctx.Push(&operation.Mul[T]{
								Scalar: din,
								A:      d,
								B:      in,
							})

							ops = append(ops, din)
						}
					}

					dw := &number.Scalar[T]{}
					ctx.Push(&operation.Sum[T]{
						Scalar: dw,
						Args:   ops,
					})
					ctx.Push(&operation.Sum[T]{
						Scalar: nw,
						Args:   []*number.Scalar[T]{w, dw, mw},
					})
					ctx.Push(&operation.Set[T]{
						Scalar: mw,
						O:      dw,
					})
				}
			}
		}
	}

	return p.PostBuildBackpropagation(ctx, Alpha, Momentum)
}

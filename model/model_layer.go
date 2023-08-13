package model

import (
	"io"
	"strings"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
)

func (p *Model[T]) Build() (uint, error) {
	return p.LastLayer.Build()
}

func (p *Model[T]) BuildFeedforward(ctx *context.Context) error {
	return p.LastLayer.BuildFeedforward(ctx)
}

func (p *Model[T]) BuildBackpropagation(
	ctx *context.Context,
	a, m *number.Scalar[T],
) error {
	return p.LastLayer.BuildBackpropagation(ctx, a, m)
}

func (p *Model[T]) Reset(ctx *context.Context) error {
	return p.LastLayer.Reset(ctx)
}

func (p *Model[T]) ResetFit(ctx *context.Context) error {
	return p.LastLayer.ResetFit(ctx)
}

func (p *Model[T]) Fit() error {
	return p.LastLayer.Fit()
}

func (p *Model[T]) SetTrainable(t bool) {
	cur := p.LastLayer
	for cur != nil && cur != p.FirstLayer {
		cur.SetTrainable(t)
		cur = cur.GetPrelayer()
	}
}

func (p *Model[T]) GetInputs() tensor.Tensor[T] {
	return p.FirstLayer.GetInputs()
}

func (p *Model[T]) GetOutputs() tensor.Tensor[T] {
	return p.LastLayer.GetOutputs()
}

func (p *Model[T]) GetNetas() tensor.Tensor[T] {
	return p.LastLayer.GetNetas()
}

func (p *Model[T]) GetDif() tensor.Tensor[T] {
	return p.LastLayer.GetDif()
}

func (p *Model[T]) GetRef() *number.Scalar[T] {
	return p.LastLayer.GetRef()
}

func (p *Model[T]) GetActivation() activation.Activation[T] {
	return p.LastLayer.GetActivation()
}

func (p *Model[T]) GetPrelayer() layer.Layer[T] {
	return p.FirstLayer.GetPrelayer()
}

func (p *Model[T]) Connect(l layer.Layer[T]) {
	p.FirstLayer.Connect(l)
}

func (p *Model[T]) PushToString(sb *strings.Builder) {
	p.LastLayer.PushToString(sb)
}

func (p *Model[T]) ResetPrinted() {
	p.LastLayer.ResetPrinted()
}

func (p *Model[T]) GetIndex() uint {
	return p.LastLayer.GetIndex()
}

func (p *Model[T]) Load(r io.Reader) error {
	return p.LastLayer.Load(r)
}

func (p *Model[T]) Save(w io.Writer) error {
	return p.LastLayer.Save(w)
}

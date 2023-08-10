package builder

import (
	"github.com/julioguillermo/neuralnetwork/v2/layer"
	"github.com/julioguillermo/neuralnetwork/v2/model"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type ModelBuilder[T types.Number] struct {
	first layer.Layer[T]
	last  layer.Layer[T]
}

func NewModelBuilder[T types.Number]() *ModelBuilder[T] {
	return &ModelBuilder[T]{}
}

func (p *ModelBuilder[T]) Push(layer layer.Layer[T]) *ModelBuilder[T] {
	if p.last == nil {
		p.first = layer
		p.last = layer
	} else {
		layer.Conect(p.last)
		p.last = layer
	}
	return p
}

func (p *ModelBuilder[T]) BuildModel() (*model.Model[T], error) {
	return model.FromInOut[T](p.first, p.last)
}

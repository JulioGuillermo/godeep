package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Sum[T types.Number] struct {
	*number.Scalar[T]
	Args []*number.Scalar[T]
}

func (p *Sum[T]) Cal() {
	// scalar := p.Args[0].Get()
	scalar := p.Args[0].Value
	size := len(p.Args)
	for i := 1; i < size; i++ {
		// scalar += p.Args[i].Get()
		scalar += p.Args[i].Value
	}
	p.Value = scalar
	// p.Set(scalar)
}

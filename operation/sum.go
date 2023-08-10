package operation

import "github.com/julioguillermo/neuralnetwork/v2/types"

type Sum[T types.Number] struct {
	*Operand[T]
	Args []*Operand[T]
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

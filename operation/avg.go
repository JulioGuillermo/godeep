package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Avg[T types.Number] struct {
	*number.Scalar[T]
	Args []*number.Scalar[T]
}

func (p *Avg[T]) Cal() {
	// num := p.Args[0].Get()
	num := p.Args[0].Value
	size := len(p.Args)
	for i := 1; i < size; i++ {
		// num += p.Args[i].Get()
		num += p.Args[i].Value
	}
	// p.Set(num / T(size))
	p.Value = num / T(size)
}

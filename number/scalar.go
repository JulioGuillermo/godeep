package number

import "github.com/julioguillermo/godeep/types"

type Scalar[T types.Number] struct {
	Value T
	// mutex sync.Mutex
}

//func (p *Operand[_]) Lock() {
//	p.mutex.Lock()
//}
//
//func (p *Operand[_]) Unlock() {
//	p.mutex.Unlock()
//}

//func (p *Operand[T]) Set(s T) {
//	p.Value = s
//	// p.mutex.Unlock()
//}
//
//func (p *Operand[T]) Get() T {
//	// p.mutex.Lock()
//	// defer p.mutex.Unlock()
//	return p.Value
//}

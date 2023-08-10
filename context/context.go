package context

import "github.com/julioguillermo/neuralnetwork/v2/operation"

type node struct {
	op   operation.Operation
	next *node
}

type Context struct {
	size  uint
	start *node
	end   *node
}

func NewCTX() *Context {
	return &Context{}
}

func (p *Context) Push(o operation.Operation) {
	p.size++
	n := &node{
		op: o,
	}
	if p.start == nil {
		p.start = n
		p.end = n
	} else {
		p.end.next = n
		p.end = n
	}
}

func (p *Context) GetOps() []operation.Operation {
	ops := make([]operation.Operation, p.size)
	cur := p.start
	for i := range ops {
		ops[i] = cur.op
		cur = cur.next
	}
	return ops
}

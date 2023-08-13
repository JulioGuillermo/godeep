package tools_test

import (
	"testing"

	"github.com/julioguillermo/godeep/tools"
)

func TestUniDimIndex(t *testing.T) {
	shape := []uint{9}
	mul := tools.GetIndexMul(shape)

	index := []uint{7}
	t.Log(index)
	ind, err := tools.GetIndex(mul, shape, index)
	t.Log(ind)
	if err != nil {
		t.Fatal(err)
	}

	idx := tools.ReverseIndex(mul, shape, ind)
	t.Log(idx)

	err = tools.GetEqShapeErr("Testing get index and reverse index", index, idx)
	if err != nil {
		t.Fatal(err)
	}
}

func Test2DimIndex(t *testing.T) {
	shape := []uint{9, 10}
	mul := tools.GetIndexMul(shape)

	index := []uint{7, 8}
	t.Log(index)
	ind, err := tools.GetIndex(mul, shape, index)
	t.Log(ind)
	if err != nil {
		t.Fatal(err)
	}

	idx := tools.ReverseIndex(mul, shape, ind)
	t.Log(idx)

	err = tools.GetEqShapeErr("Testing get index and reverse index", index, idx)
	if err != nil {
		t.Fatal(err)
	}
}

func TestMultiDimIndex(t *testing.T) {
	shape := []uint{9, 8, 4, 6}
	mul := tools.GetIndexMul(shape)

	index := []uint{7, 1, 2, 0}
	t.Log(index)
	ind, err := tools.GetIndex(mul, shape, index)
	t.Log(ind)
	if err != nil {
		t.Fatal(err)
	}

	idx := tools.ReverseIndex(mul, shape, ind)
	t.Log(idx)

	err = tools.GetEqShapeErr("Testing get index and reverse index", index, idx)
	if err != nil {
		t.Fatal(err)
	}
}

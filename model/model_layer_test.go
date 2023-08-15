package model_test

import (
	"testing"

	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/model"
)

func TestModelAsLayer(t *testing.T) {
	var m layer.Layer[float32] = model.NewModel[float32]()
	t.Log(m)
}

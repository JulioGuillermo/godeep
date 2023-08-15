package img

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

const MaxColor = 65535

func LoadImage[T types.Number](path string) (tensor.Tensor[T], error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	size := img.Bounds().Size()
	width := size.X
	height := size.Y

	t := tensor.NewZeros[T](4, uint(width), uint(height))
	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			c := img.At(i, j)
			r, g, b, a := c.RGBA()
			t.Set(T(float64(r)/MaxColor), 0, uint(i), uint(j))
			t.Set(T(float64(g)/MaxColor), 1, uint(i), uint(j))
			t.Set(T(float64(b)/MaxColor), 2, uint(i), uint(j))
			t.Set(T(float64(a)/MaxColor), 3, uint(i), uint(j))
		}
	}

	return t, nil
}

func LoadSimpleScaled[T types.Number](path string, width, height uint) (tensor.Tensor[T], error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	size := img.Bounds().Size()

	scalX := float64(size.X) / float64(width)
	scalY := float64(size.Y) / float64(height)
	t := tensor.NewZeros[T](4, width, height)

	var i uint
	var j uint
	var X int
	var Y int
	for i = 0; i < width; i++ {
		for j = 0; j < height; j++ {
			X = int(float64(i) * scalX)
			Y = int(float64(j) * scalY)
			c := img.At(X, Y)
			r, g, b, a := c.RGBA()
			t.Set(T(float64(r)/MaxColor), 0, uint(i), uint(j))
			t.Set(T(float64(g)/MaxColor), 1, uint(i), uint(j))
			t.Set(T(float64(b)/MaxColor), 2, uint(i), uint(j))
			t.Set(T(float64(a)/MaxColor), 3, uint(i), uint(j))
		}
	}

	return t, nil
}

func LoadSmoothScaled[T types.Number](
	path string,
	width, height uint,
) (tensor.Tensor[T], error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	size := img.Bounds().Size()

	scaleX := float64(size.X) / float64(width)
	scaleY := float64(size.Y) / float64(height)
	t := tensor.NewZeros[T](4, width, height)

	// Realiza el escalado suave de la imagen
	for x := uint(0); x < width; x++ {
		for y := uint(0); y < height; y++ {
			// Calcula las coordenadas correspondientes en la imagen original
			srcX := int(float64(x) * scaleX)
			srcY := int(float64(y) * scaleY)

			// Obtiene los píxeles vecinos
			p1 := img.At(srcX, srcY)
			p2 := img.At(srcX+1, srcY)
			p3 := img.At(srcX, srcY+1)
			p4 := img.At(srcX+1, srcY+1)

			// Calcula los pesos de interpolación
			dx := float64(x)*scaleX - float64(srcX)
			dy := float64(y)*scaleY - float64(srcY)

			// Realiza la interpolación bilineal suave
			r1, g1, b1, a1 := p1.RGBA()
			r2, g2, b2, a2 := p2.RGBA()
			r3, g3, b3, a3 := p3.RGBA()
			r4, g4, b4, a4 := p4.RGBA()

			r := uint32(
				(1.0-dx)*(1.0-dy)*float64(
					r1,
				) + dx*(1.0-dy)*float64(
					r2,
				) + (1.0-dx)*dy*float64(
					r3,
				) + dx*dy*float64(
					r4,
				),
			)
			g := uint32(
				(1.0-dx)*(1.0-dy)*float64(
					g1,
				) + dx*(1.0-dy)*float64(
					g2,
				) + (1.0-dx)*dy*float64(
					g3,
				) + dx*dy*float64(
					g4,
				),
			)
			b := uint32(
				(1.0-dx)*(1.0-dy)*float64(
					b1,
				) + dx*(1.0-dy)*float64(
					b2,
				) + (1.0-dx)*dy*float64(
					b3,
				) + dx*dy*float64(
					b4,
				),
			)
			a := uint32(
				(1.0-dx)*(1.0-dy)*float64(
					a1,
				) + dx*(1.0-dy)*float64(
					a2,
				) + (1.0-dx)*dy*float64(
					a3,
				) + dx*dy*float64(
					a4,
				),
			)

			// Asigna el nuevo color al píxel en la imagen escalada
			t.Set(T(float64(r)/MaxColor), 0, uint(x), uint(y))
			t.Set(T(float64(g)/MaxColor), 1, uint(x), uint(y))
			t.Set(T(float64(b)/MaxColor), 2, uint(x), uint(y))
			t.Set(T(float64(a)/MaxColor), 3, uint(x), uint(y))
		}
	}

	return t, nil
}

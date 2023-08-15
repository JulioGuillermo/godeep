package img

//func smoothResize(src image.Image, newWidth, newHeight int) image.Image {
//    // Crea una nueva imagen con el tamaño especificado
//    dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
//
//    // Calcula las proporciones de escalado en X e Y
//    scaleX := float64(src.Bounds().Dx()) / float64(newWidth)
//    scaleY := float64(src.Bounds().Dy()) / float64(newHeight)
//
//    // Realiza el escalado suave de la imagen
//    for y := 0; y < newHeight; y++ {
//        for x := 0; x < newWidth; x++ {
//            // Calcula las coordenadas correspondientes en la imagen original
//            srcX := int(float64(x) * scaleX)
//            srcY := int(float64(y) * scaleY)
//
//            // Obtiene los píxeles vecinos
//            p1 := src.At(srcX, srcY)
//            p2 := src.At(srcX+1, srcY)
//            p3 := src.At(srcX, srcY+1)
//            p4 := src.At(srcX+1, srcY+1)
//
//            // Calcula los pesos de interpolación
//            dx := float64(x)*scaleX - float64(srcX)
//            dy := float64(y)*scaleY - float64(srcY)
//
//            // Realiza la interpolación bilineal suave
//            r1, g1, b1, a1 := p1.RGBA()
//            r2, g2, b2, a2 := p2.RGBA()
//            r3, g3, b3, a3 := p3.RGBA()
//            r4, g4, b4, a4 := p4.RGBA()
//
//            r := uint32((1.0-dx)*(1.0-dy)*float64(r1) + dx*(1.0-dy)*float64(r2) + (1.0-dx)*dy*float64(r3) + dx*dy*float64(r4))
//            g := uint32((1.0-dx)*(1.0-dy)*float64(g1) + dx*(1.0-dy)*float64(g2) + (1.0-dx)*dy*float64(g3) + dx*dy*float64(g4))
//            b := uint32((1.0-dx)*(1.0-dy)*float64(b1) + dx*(1.0-dy)*float64(b2) + (1.0-dx)*dy*float64(b3) + dx*dy*float64(b4))
//            a := uint32((1.0-dx)*(1.0-dy)*float64(a1) + dx*(1.0-dy)*float64(a2) + (1.0-dx)*dy*float64(a3) + dx*dy*float64(a4))
//
//            // Asigna el nuevo color al píxel en la imagen escalada
//            dst.Set(x, y, color.RGBA{
//                R: uint8(r >> 8),
//                G: uint8(g >> 8),
//                B: uint8(b >> 8),
//                A: uint8(a >> 8),
//            })
//        }
//    }
//
//    return dst
//}

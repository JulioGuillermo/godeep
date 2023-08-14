package tools

import "strings"

func Bar(pro float64, size int) string {
	if size <= 0 {
		return ""
	}
	if size < 4 {
		return strings.Repeat(".", size)
	}
	size -= 2
	prog := int(float64(size) * pro)
	left := size - prog
	if left < 0 {
		left = 0
	}

	var sb strings.Builder
	sb.WriteRune('[')
	sb.WriteString(strings.Repeat("=", prog))
	sb.WriteString(strings.Repeat(" ", left))
	sb.WriteRune(']')
	return sb.String()
}

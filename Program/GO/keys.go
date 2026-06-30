package main

import "sync"

// KeyManager は複数のAPIキーを並列ワーカー間で安全にローテーションするための構造体．
// レート制限(429)を検知したワーカーがRotateを呼ぶと，以後Current()を呼ぶワーカーは
// 次のキーを使うようになる．
type KeyManager struct {
	mu   sync.Mutex
	keys []string
	idx  int
}

func NewKeyManager(keys []string) *KeyManager {
	return &KeyManager{keys: keys}
}

// Current は現在選択されているAPIキーを返す．
func (k *KeyManager) Current() string {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.keys[k.idx]
}

// Rotate は次のAPIキーに切り替えて，切替後のキーを返す．
func (k *KeyManager) Rotate() string {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.idx = (k.idx + 1) % len(k.keys)
	return k.keys[k.idx]
}

func (k *KeyManager) Len() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	return len(k.keys)
}

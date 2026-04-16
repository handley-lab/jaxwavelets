# Maintainer: Will Handley <wh260@cam.ac.uk>
pkgname=python-jaxwt
pkgver=0.1.0
pkgrel=1
pkgdesc="JAX-native wavelet transforms"
arch=('any')
url="https://github.com/handley-lab/jaxwt"
license=('MIT')
depends=('python' 'python-jax')
makedepends=('python-build' 'python-installer' 'python-setuptools')
checkdepends=('python-pytest' 'python-pywavelets')
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
    cd jaxwt-$pkgver
    python -m build --wheel --no-isolation
}

check() {
    cd jaxwt-$pkgver
    JAX_ENABLE_X64=1 python -m pytest jaxwt/tests/ -x --tb=short
}

package() {
    cd jaxwt-$pkgver
    python -m installer --destdir="$pkgdir" dist/*.whl
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}

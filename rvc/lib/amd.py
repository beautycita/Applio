import torch

# MIOpen has no usable dilated 1D convolution kernel on some AMD architectures. On gfx1100 the
# identical FLOPs run ~30x slower dilated than undilated, and the HiFi-GAN / NSF ResBlocks are built
# on dilations (1, 3, 5), so this dominates both training and inference.
#
# A dilation-D convolution is exactly D independent undilated convolutions over the strided phases
# x[..., i::D], which avoids the kernel entirely. Whether that wins is a property of the card and
# the ROCm build rather than of the vendor, so it is measured once here at startup and the native
# kernel keeps ties. Patching F.conv1d rather than the models means every dilated conv is covered,
# including ones outside the ResBlocks.
if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name():
    import time

    _conv1d = torch.nn.functional.conv1d

    def _conv1d_phases(input, weight, bias, pad, dilation, groups):
        length = input.shape[-1]
        input = torch.nn.functional.pad(input, (pad, pad))
        # phases must divide evenly; the remainder is padded off the end, so it only ever lands
        # beyond `length` and is dropped by the final slice
        rem = (-input.shape[-1]) % dilation
        if rem:
            input = torch.nn.functional.pad(input, (0, rem))
        n, c, total = input.shape
        phases = (
            input.view(n, c, total // dilation, dilation)
            .permute(0, 3, 1, 2)
            .reshape(n * dilation, c, total // dilation)
        )
        out = _conv1d(phases, weight, bias, 1, 0, 1, groups)
        chan, per = weight.shape[0], out.shape[-1]
        out = (
            out.view(n, dilation, chan, per)
            .permute(0, 2, 3, 1)
            .reshape(n, chan, per * dilation)
        )
        return out[..., :length]

    def z_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        first = lambda v: v[0] if isinstance(v, (tuple, list)) else v
        d, s, p = first(dilation), first(stride), first(padding)
        # only the plain dilated case is rewritten; everything else falls through untouched
        if d == 1 or s != 1 or not isinstance(p, int) or not input.is_cuda:
            return _conv1d(input, weight, bias, stride, padding, dilation, groups)
        return _conv1d_phases(input, weight, bias, p, d, groups)

    def _dilated_conv_is_slow():
        conv = torch.nn.Conv1d(192, 192, 3, dilation=3, padding=3).cuda().eval()
        x = torch.randn(4, 192, 4096, device="cuda")

        def timed(fn):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        with torch.no_grad():
            native = timed(lambda: _conv1d(x, conv.weight, conv.bias, 1, 3, 3, 1))
            phases = timed(lambda: _conv1d_phases(x, conv.weight, conv.bias, 3, 3, 1))
        # a clear margin, so timing noise cannot pick the rewrite on a card whose kernel is fine
        return phases * 1.5 < native

    try:
        if _dilated_conv_is_slow():
            torch.nn.functional.conv1d = z_conv1d
    except Exception:
        pass  # any failure here just leaves the stock kernel in use

#include "sky_renderer.h"
#include "indicom.h"
#include "locale_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
// Build a normalized 1D Gaussian kernel of the given sigma, truncated at 'radius' taps each side.
std::vector<float> buildGaussianKernel1D(float sigma, int radius)
{
    std::vector<float> kernel(2 * radius + 1);
    float const inv2sig2 = sigma > 0.0f ? 1.0f / (2.0f * sigma * sigma) : 0.0f;
    float sum = 0.0f;

    for (int i = -radius; i <= radius; i++)
    {
        float const w = sigma > 0.0f ? std::exp(-static_cast<float>(i * i) * inv2sig2) : (i == 0 ? 1.0f : 0.0f);
        kernel[i + radius] = w;
        sum += w;
    }
    if (sum > 0.0f)
        for (float &w : kernel)
            w /= sum;

    return kernel;
}

// Separable 2D Gaussian blur of a square buffer, in place. Out-of-range taps are
// skipped rather than clamped, i.e. the buffer is implicitly zero-padded at its edges.
void blurSquareBuffer(std::vector<float> &buf, int size,
                      const std::vector<float> &kernelX, const std::vector<float> &kernelY)
{
    int const radiusX = static_cast<int>(kernelX.size() / 2);
    int const radiusY = static_cast<int>(kernelY.size() / 2);

    std::vector<float> tmp(buf.size(), 0.0f);

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float acc = 0.0f;
            for (int k = -radiusX; k <= radiusX; k++)
            {
                int const sx = x + k;
                if (sx < 0 || sx >= size)
                    continue;
                acc += buf[y * size + sx] * kernelX[k + radiusX];
            }
            tmp[y * size + x] = acc;
        }
    }

    for (int x = 0; x < size; x++)
    {
        for (int y = 0; y < size; y++)
        {
            float acc = 0.0f;
            for (int k = -radiusY; k <= radiusY; k++)
            {
                int const sy = y + k;
                if (sy < 0 || sy >= size)
                    continue;
                acc += tmp[sy * size + x] * kernelY[k + radiusY];
            }
            buf[y * size + x] = acc;
        }
    }
}
} // namespace

double SkyRenderer::flux(double mag) const
{
    if (m_Cfg.limitingMag == m_Cfg.saturationMag)
        return 1.0;
    double const z = m_Cfg.limitingMag;
    double const k = 2.5 * log10(m_Cfg.maxVal) / (m_Cfg.limitingMag - m_Cfg.saturationMag);
    return pow(10.0, (z - mag) * k / 2.5);
}

int SkyRenderer::addToPixel(INDI::CCDChip *chip, int x, int y, int val)
{
    int const nwidth  = chip->getSubW();
    int const nheight = chip->getSubH();

    x -= chip->getSubX();
    y -= chip->getSubY();

    if (x < 0 || x >= nwidth || y < 0 || y >= nheight)
        return 0;

    auto *pt = reinterpret_cast<uint16_t *>(chip->getFrameBuffer());
    pt += y * nwidth + x;

    int newval = static_cast<int>(pt[0]) + val;
    if (newval > m_Cfg.maxVal)
        newval = m_Cfg.maxVal;
    if (newval > m_MaxPix)
        m_MaxPix = newval;
    if (newval < m_MinPix)
        m_MinPix = newval;
    pt[0] = static_cast<uint16_t>(newval);

    return 1;
}

int SkyRenderer::drawImageStar(INDI::CCDChip *chip, float mag, float x, float y, float exp_s)
{
    // Donut simulation is opt-in (obstruction ratio > 0); when off, fall through to
    // the plain Gaussian PSF below, unchanged from its original behavior.
    if (m_Cfg.donutObstruction > 0.0f)
        return drawDonutStar(chip, mag, x, y, exp_s);

    int drew = 0;

    int const subX = chip->getSubX();
    int const subY = chip->getSubY();
    int const subW = chip->getSubW() + subX;
    int const subH = chip->getSubH() + subY;

    if (x < subX || x > subW || y < subY || y > subH)
        return 0;

    float const totalFlux = static_cast<float>(flux(mag) * exp_s);

    // Compute position-dependent seeing to simulate sensor tilt.
    // Instead of a global seeing, compute per-star seeing by shifting the
    // effective focus position based on star location on the sensor.
    // This causes each tile's V-curve minimum to land at a different focuser
    // position — which is exactly what the Aberration Inspector detects as tilt.
    float effectiveSeeing = m_Cfg.seeing;
    if (m_Cfg.tiltLR != 0.0f || m_Cfg.tiltTB != 0.0f)
    {
        float const cx = chip->getXRes() / 2.0f;
        float const cy = chip->getYRes() / 2.0f;
        float const dx = (x - cx) / cx;  // normalized [-1, +1]
        float const dy = (y - cy) / cy;
        // tiltLR/tiltTB in arcsec: directly added to seeing at that position.
        // This shifts where the V-curve minimum occurs for edge tiles vs center.
        float const tiltOffset = m_Cfg.tiltLR * dx + m_Cfg.tiltTB * dy;
        effectiveSeeing += tiltOffset;
        if (effectiveSeeing < 0.5f)
            effectiveSeeing = 0.5f;  // Floor to prevent degenerate PSF
    }

    int const boxsizey = static_cast<int>(3.0f * effectiveSeeing / m_ImageScaleY) + 1;

    float const sigma    = effectiveSeeing / (2.0f * std::sqrt(2.0f * std::log(2.0f)));
    float const norm     = 1.0f / (sigma * std::sqrt(2.0f * float(M_PI)));
    float const inv2sig2 = 1.0f / (2.0f * sigma * sigma);

    // Sub-pixel centering: keep the fractional part of the star position so the
    // PSF is centred on the true (floating-point) location rather than snapped to
    // the truncated integer pixel. Without this, sub-pixel drift is discarded and
    // the measured centroid quantises to a lattice (see indilib/indi#2465).
    float const fracX = x - static_cast<int>(x);
    float const fracY = y - static_cast<int>(y);

    for (int sy = -boxsizey; sy <= boxsizey; sy++)
    {
        for (int sx = -boxsizey; sx <= boxsizey; sx++)
        {
            float const ddx = m_ImageScaleX * (sx - fracX);
            float const ddy = m_ImageScaleY * (sy - fracY);
            float const dc2 = ddx * ddx + ddy * ddy;
            float const fa  = norm * std::exp(-dc2 * inv2sig2);
            int const fp = static_cast<int>(fa * totalFlux);
            if (fp > 0)
            {
                if (addToPixel(chip,
                               static_cast<int>(x) + sx,
                               static_cast<int>(y) + sy, fp) != 0)
                    drew = 1;
            }
        }
    }
    return drew;
}

// Render a defocused "donut" star via geometric ray sampling of the mirror pupil,
// softened by a real convolution with the seeing kernel. Used to test the Collimator
// module against a known, controllable ground truth (collimation error, field coma)
// instead of a plain Gaussian PSF.
int SkyRenderer::drawDonutStar(INDI::CCDChip *chip, float mag, float x, float y, float exp_s)
{
    int const subX = chip->getSubX();
    int const subY = chip->getSubY();
    int const subW = chip->getSubW() + subX;
    int const subH = chip->getSubH() + subY;

    if (x < subX || x > subW || y < subY || y > subH)
        return 0;

    float const totalFlux = static_cast<float>(flux(mag) * exp_s);

    // Geometric radius of the mirror's shadow (the "pupil"), growing linearly with
    // distance from focus -- this is what turns a defocused star into a donut, well
    // before diffraction effects would matter.
    float const rOuterArcsec = m_Cfg.donutDefocusSlope * std::fabs(m_Cfg.donutTicks);
    float const rInnerArcsec = rOuterArcsec * m_Cfg.donutObstruction;

    // Collimation error: a fixed shift of the secondary's shadow relative to the
    // primary aperture, independent of the star's field position. It only affects
    // which rays are physically blocked (see the exclusion test below), not where an
    // unblocked ray lands -- a tilted secondary doesn't move the star itself. It
    // flips sign between intra- and extra-focal, unlike field coma below.
    float const collimSign = m_Cfg.donutTicks >= 0.0f ? 1.0f : -1.0f;
    float const shadowOffsetX = m_Cfg.donutCollimDx * collimSign;
    float const shadowOffsetY = m_Cfg.donutCollimDy * collimSign;

    // Field coma: displaces each pupil zone (radius rho, 0=center, 1=edge) by an
    // amount growing with rho^2, with the classic Seidel angular pattern relative to
    // the star's own radial direction in the field ("2 + cos2*theta'" along that
    // direction, "sin2*theta'" across it). Unlike collimation error, this is applied
    // to every ray's landing position, not just to the obstruction test -- rays near
    // the obstruction (small rho) are barely affected, so the hole stays roughly
    // centered, while the outer rim (rho near 1) gets pulled toward the image center
    // and bunches up on one side. That's what produces the comet/seagull/"V" shape a
    // simple decentered ring can never reproduce, no matter its offset.
    float comaScale = 0.0f;
    float phi = 0.0f; // direction from the star toward the image center
    if (m_Cfg.donutComaCoefficient != 0.0f && m_Cfg.fRatio > 0.0f)
    {
        float const cx = chip->getXRes() / 2.0f;
        float const cy = chip->getYRes() / 2.0f;
        float const fieldDx = x - cx;
        float const fieldDy = y - cy;
        float const fieldRadius = std::sqrt(fieldDx * fieldDx + fieldDy * fieldDy);
        float const halfDiag = std::sqrt(cx * cx + cy * cy);

        if (fieldRadius > 0.0f && halfDiag > 0.0f)
        {
            comaScale = m_Cfg.donutComaCoefficient * (fieldRadius / halfDiag) / (m_Cfg.fRatio * m_Cfg.fRatio);
            // Empirically flipped 180 deg from the naive "toward center" vector: the
            // Seidel coma sign convention placed the flare on the wrong side.
            phi = std::atan2(fieldDy, fieldDx);
        }
    }

    // Blur kernel: the constant atmospheric/optical seeing, converted from FWHM to
    // sigma and from arcsec to pixels. Because this is a real convolution rather than
    // an edge-softening shortcut, it naturally reduces to the plain seeing PSF as the
    // pupil shrinks toward focus.
    float const sigmaArcsec = m_Cfg.donutBaseSeeing / (2.0f * std::sqrt(2.0f * std::log(2.0f)));
    float const sigmaX = sigmaArcsec / m_ImageScaleX;
    float const sigmaY = sigmaArcsec / m_ImageScaleY;
    int const kernelRadiusX = std::max(1, static_cast<int>(std::ceil(3.0f * sigmaX)));
    int const kernelRadiusY = std::max(1, static_cast<int>(std::ceil(3.0f * sigmaY)));
    int const kernelRadius = std::max(kernelRadiusX, kernelRadiusY);

    // Box sizing: cover the widest possible ray excursion (defocus radius plus the
    // largest possible coma displacement) with margin for the blur kernel.
    float const maxExtentArcsec = rOuterArcsec + 3.0f * comaScale;
    int const outerPx = static_cast<int>(std::ceil(maxExtentArcsec / std::min(m_ImageScaleX, m_ImageScaleY)));
    int const boxRadius = outerPx + kernelRadius + 1;
    int const boxSize = 2 * boxRadius + 1;

    // Sub-pixel centering: keep the fractional part of the star position, same as the
    // plain Gaussian path (see indilib/indi#2465).
    float const fracX = x - static_cast<int>(x);
    float const fracY = y - static_cast<int>(y);

    std::vector<float> mask(static_cast<size_t>(boxSize) * boxSize, 0.0f);
    float maskSum = 0.0f;

    // Geometric ray sampling: walk the pupil in polar coordinates (rho, theta) and,
    // for each unobstructed ray, splat its image-plane landing point into the mask,
    // weighted by rho (the pupil area element) so the outer, larger-area zones
    // contribute proportionally more flux than the inner ones, as real uniform
    // illumination would.
    int const nTheta = std::clamp(static_cast<int>(std::ceil(2.0f * static_cast<float>(M_PI) * outerPx)), 64, 2000);
    int const nRho   = std::clamp(outerPx / 3, 8, 64);

    for (int ir = 0; ir < nRho; ir++)
    {
        float const rho = (static_cast<float>(ir) + 0.5f) / static_cast<float>(nRho);

        for (int it = 0; it < nTheta; it++)
        {
            float const theta = 2.0f * static_cast<float>(M_PI) * static_cast<float>(it) / static_cast<float>(nTheta);

            // Nominal (defocus-only) landing position -- used only to test whether
            // this ray is physically blocked by the (collimation-shifted) secondary.
            float const p0x = rOuterArcsec * rho * std::cos(theta);
            float const p0y = rOuterArcsec * rho * std::sin(theta);

            float const dHoleX = p0x - shadowOffsetX;
            float const dHoleY = p0y - shadowOffsetY;
            if (std::sqrt(dHoleX * dHoleX + dHoleY * dHoleY) < rInnerArcsec)
                continue;

            // Actual landing position: defocus plus the coma term above.
            float landX = p0x;
            float landY = p0y;

            if (comaScale != 0.0f)
            {
                float const thetaLocal = theta - phi;
                float const comaLocalX = comaScale * rho * rho * (2.0f + std::cos(2.0f * thetaLocal));
                float const comaLocalY = comaScale * rho * rho * std::sin(2.0f * thetaLocal);
                landX += comaLocalX * std::cos(phi) - comaLocalY * std::sin(phi);
                landY += comaLocalX * std::sin(phi) + comaLocalY * std::cos(phi);
            }

            // Bilinear splat into the mask grid.
            float const gridX = landX / m_ImageScaleX + fracX + boxRadius;
            float const gridY = landY / m_ImageScaleY + fracY + boxRadius;
            int const ix0 = static_cast<int>(std::floor(gridX));
            int const iy0 = static_cast<int>(std::floor(gridY));
            float const fx = gridX - static_cast<float>(ix0);
            float const fy = gridY - static_cast<float>(iy0);

            int const cornerDx[4] = { 0, 1, 0, 1 };
            int const cornerDy[4] = { 0, 0, 1, 1 };
            float const cornerW[4] = { (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy };

            for (int c = 0; c < 4; c++)
            {
                int const px = ix0 + cornerDx[c];
                int const py = iy0 + cornerDy[c];
                if (px < 0 || px >= boxSize || py < 0 || py >= boxSize)
                    continue;
                float const contribution = rho * cornerW[c];
                mask[py * boxSize + px] += contribution;
                maskSum += contribution;
            }
        }
    }

    if (maskSum <= 0.0f)
        return 0;

    std::vector<float> const kernelX = buildGaussianKernel1D(sigmaX, kernelRadiusX);
    std::vector<float> const kernelY = buildGaussianKernel1D(sigmaY, kernelRadiusY);
    blurSquareBuffer(mask, boxSize, kernelX, kernelY);

    float blurredSum = 0.0f;
    for (float const v : mask)
        blurredSum += v;

    if (blurredSum <= 0.0f)
        return 0;

    // Rescale so the blurred shape carries the star's true total flux -- the mask
    // started as a 0/1 pattern, not an analytically normalized profile.
    float const scale = totalFlux / blurredSum;

    int drew = 0;
    for (int sy = -boxRadius; sy <= boxRadius; sy++)
    {
        for (int sx = -boxRadius; sx <= boxRadius; sx++)
        {
            float const v = mask[(sy + boxRadius) * boxSize + (sx + boxRadius)];
            int const fp = static_cast<int>(v * scale);
            if (fp > 0)
            {
                if (addToPixel(chip, static_cast<int>(x) + sx, static_cast<int>(y) + sy, fp) != 0)
                    drew = 1;
            }
        }
    }
    return drew;
}

void SkyRenderer::applyReadoutNoise(INDI::CCDChip *chip, int bias, int maxNoise)
{
    if (maxNoise <= 0)
        return;

    int const nx = chip->getSubW();
    int const ny = chip->getSubH();
    auto *buf = reinterpret_cast<uint16_t *>(chip->getFrameBuffer());

    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            int newval = static_cast<int>(buf[y * nx + x]) + bias + (random() % maxNoise);
            if (newval > m_Cfg.maxVal)
                newval = m_Cfg.maxVal;
            buf[y * nx + x] = static_cast<uint16_t>(newval);
        }
}

void SkyRenderer::drawSkyGlow(INDI::CCDChip *chip, float exp_s)
{
    float const skyflux = static_cast<float>(flux(m_Cfg.skyGlow)) * exp_s;

    int const nwidth  = chip->getSubW();
    int const nheight = chip->getSubH();

    auto *pt = reinterpret_cast<uint16_t *>(chip->getFrameBuffer());

    float const vig = std::min(nwidth, nheight) * m_ImageScaleX;
    float const invVig2 = 1.0f / (vig * vig);

    for (int y = 0; y < nheight; y++)
    {
        float const sy = nheight / 2.0f - y;

        for (int x = 0; x < nwidth; x++)
        {
            float const sx = nwidth / 2.0f - x;

            float const dc2 = sx * sx * m_ImageScaleX * m_ImageScaleX
                              + sy * sy * m_ImageScaleY * m_ImageScaleY;

            float const fa = std::exp(-2.0f * 0.7f * dc2 * invVig2);

            float fp = (pt[0] + skyflux) * fa;

            if (fp > m_Cfg.maxVal) fp = static_cast<float>(m_Cfg.maxVal);
            if (fp < pt[0]) fp = pt[0]; // never darken an already-bright pixel
            if (fp > m_MaxPix) m_MaxPix = static_cast<int>(fp);
            if (fp < m_MinPix) m_MinPix = static_cast<int>(fp);

            pt[0] = static_cast<uint16_t>(fp);
            pt++;
        }
    }
}

int SkyRenderer::renderFrame(
    INDI::CCDChip *chip,
    double ra_j2000_deg,
    double dec_j2000_deg,
    double focal_length_mm,
    double rotation_deg,
    float  exposure_s,
    bool   renderStars,
    double minSearchRadiusArcmin)
{
    m_MaxPix = 0;
    m_MinPix = 65000;

    // Image scale (arcsec/pixel) from focal length and pixel size
    m_ImageScaleX = static_cast<float>((chip->getPixelSizeX() / focal_length_mm) * 206.3);
    m_ImageScaleY = static_cast<float>((chip->getPixelSizeY() / focal_length_mm) * 206.3);

    // Plate matrix: maps standard coords to pixel coords, with camera rotation applied
    double const theta = rotation_deg * (M_PI / 180.0);
    double const pprx  = focal_length_mm / chip->getPixelSizeX() * 1000.0; // pixels per radian, X
    double const ppry  = focal_length_mm / chip->getPixelSizeY() * 1000.0; // pixels per radian, Y

    double const pa =  pprx * std::cos(theta);
    double const pb =  ppry * std::sin(theta);
    double const pd = -pprx * std::sin(theta);
    double const pe =  ppry * std::cos(theta);
    double const pc =  chip->getXRes() / 2.0;  // X center pixel
    double const pf =  chip->getYRes() / 2.0;  // Y center pixel
    double const ccdW = chip->getXRes();

    // Field center in radians
    double const rar  = ra_j2000_deg  * (M_PI / 180.0);
    double const decr = dec_j2000_deg * (M_PI / 180.0);

    // GSC search radius (arcmin): half-diagonal of the chip in arcsec, converted to arcmin
    double radius = std::sqrt(
                        m_ImageScaleX * m_ImageScaleX * chip->getXRes() / 2.0 * chip->getXRes() / 2.0 +
                        m_ImageScaleY * m_ImageScaleY * chip->getYRes() / 2.0 * chip->getYRes() / 2.0) / 60.0;
    if (minSearchRadiusArcmin > 0 && radius < minSearchRadiusArcmin)
        radius = minSearchRadiusArcmin;

    // Cap lookup magnitude for very wide fields to limit GSC processing time
    double lookuplimit = m_Cfg.limitingMag;
    if (radius > 60)
        lookuplimit = 11;

    // Clear the frame buffer
    memset(chip->getFrameBuffer(), 0, chip->getFrameBufferSize());

    int drawn = 0;

    if (renderStars)
    {
        AutoCNumeric locale;
        char gsccmd[250];

        // Handbook of astronomical image processing, eqs. 9.1/9.2 (gnomonic projection)
        snprintf(gsccmd, sizeof(gsccmd),
                 "gsc -c %8.6f %+8.6f -r %4.1f -m 0 %4.2f -n 3000",
                 range360(ra_j2000_deg),
                 rangeDec(dec_j2000_deg),
                 radius,
                 lookuplimit);

        FILE *pp = popen(gsccmd, "r");
        if (pp != nullptr)
        {
            char line[256];

            while (fgets(line, sizeof(line), pp) != nullptr)
            {
                char  id[20], plate[6], ob[6];
                float ra, dec, pose, mag, mage, dist;
                int   band, c, dir;

                int rc = sscanf(line, "%10s %f %f %f %f %f %d %d %4s %2s %f %d",
                                id, &ra, &dec, &pose, &mag, &mage,
                                &band, &c, plate, ob, &dist, &dir);
                if (rc != 12)
                    continue;

                double const srar  = ra  * (M_PI / 180.0);
                double const sdecr = dec * (M_PI / 180.0);

                double const denom = cos(decr) * cos(sdecr) * cos(srar - rar)
                                     + sin(decr) * sin(sdecr);
                if (denom <= 0)
                    continue;

                double const sx = cos(sdecr) * sin(srar - rar) / denom;
                double const sy = (sin(decr) * cos(sdecr) * cos(srar - rar)
                                   - cos(decr) * sin(sdecr)) / denom;

                // Invert horizontally (CW->CCW, origin North)
                double const ccdx = ccdW - (pa * sx + pb * sy + pc);
                double const ccdy =          pd * sx + pe * sy + pf;

                drawn += drawImageStar(chip, mag,
                                       static_cast<float>(ccdx),
                                       static_cast<float>(ccdy),
                                       exposure_s);
            }
            pclose(pp);
        }
        else
        {
            drawn = -1;
        }
    }

    drawSkyGlow(chip, exposure_s);

    return drawn;
}

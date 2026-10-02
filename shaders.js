// ====================================================================
// 1. SHADER COMPUTE (GPGPU - Movimiento y Lógica de Physarum)
// ====================================================================
export const computeShader = `
    uniform sampler2D uTrailMap;
    uniform float uSensorDist;
    uniform float uSensorAngle;
    uniform float uRotAngle;
    uniform float uMoveDist;
    uniform vec2 uMouse;
    uniform float uAttractForce;
    uniform float uAdaptiveStrength;
    uniform float uAudioLevel;
    uniform float uMotionJitter;
    uniform float uBeat;
    uniform sampler2D uShapePositions;
    uniform float uShapeStrength;
    uniform float uTime;

    float rand(vec2 co) {
        return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }

    float getTrail(vec2 pos) {
        return texture2D(uTrailMap, fract(pos)).r;
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / resolution.xy;
        vec4 particle = texture2D(texturePosition, uv);
        
        vec2 pos = particle.xy;
        float heading = particle.z;

        // Sensado en 3 direcciones
        vec2 dirCenter = vec2(cos(heading), sin(heading));
        vec2 dirLeft   = vec2(cos(heading - uSensorAngle), sin(heading - uSensorAngle));
        vec2 dirRight  = vec2(cos(heading + uSensorAngle), sin(heading + uSensorAngle));

        float senseCenter = getTrail(pos + dirCenter * uSensorDist);
        float senseLeft   = getTrail(pos + dirLeft * uSensorDist);
        float senseRight  = getTrail(pos + dirRight * uSensorDist);
        float localDensity = clamp(getTrail(pos), 0.0, 1.0);
        float adaptive = mix(1.0, 1.0 + localDensity * 1.8, uAdaptiveStrength);

        // Reglas de decisión Physarum
        if (senseCenter > senseLeft && senseCenter > senseRight) {
            // Continuar rumbo
        } else if (senseCenter < senseLeft && senseCenter < senseRight) {
            heading += (rand(pos) > 0.5 ? 1.0 : -1.0) * uRotAngle * adaptive;
        } else if (senseLeft > senseRight) {
            heading -= uRotAngle * adaptive;
        } else if (senseRight > senseLeft) {
            heading += uRotAngle * adaptive;
        }

        // Atractor manual con interacción de puntero
        if (uAttractForce > 0.0) {
            vec2 toMouse = uMouse - pos;
            float dist = length(toMouse);
            if (dist < 0.3) {
                float targetAngle = atan(toMouse.y, toMouse.x);
                heading = mix(heading, targetAngle, uAttractForce * (1.0 - dist / 0.3));
            }
        }

        // Avanzar posición y ajustar bordes (Wrapping)
        float pulse = 1.0 + uAudioLevel * 0.12 + uBeat * 0.09;
        heading += (rand(pos + vec2(heading)) - 0.5) * (uMotionJitter + uBeat * 0.045);
        pos += vec2(cos(heading), sin(heading + 1.5708)) * uBeat * 0.0018;
        pos += vec2(cos(heading), sin(heading)) * uMoveDist * pulse / adaptive;
        pos = fract(pos + vec2(1.0));
        vec2 shapePosition = texture2D(uShapePositions, uv).xy;
        float bodyPhase = shapePosition.x * 18.0 - uTime * 2.4;
        float tailWeight = smoothstep(0.25, 0.95, abs(shapePosition.x - 0.5) * 2.0);
        shapePosition.y += sin(bodyPhase) * 0.018 * tailWeight;
        shapePosition.x += cos(bodyPhase * 0.7) * 0.005 * tailWeight;
        shapePosition.y += sin(uTime * 1.8) * 0.004;
        pos = mix(pos, shapePosition, uShapeStrength);

        gl_FragColor = vec4(pos, heading, 1.0);
    }
`;

// ====================================================================
// 2. SHADERS DE RENDERIZADO DE PARTÍCULAS (Depósito sobre el Trail Map)
// ====================================================================
export const particleVertexShader = `
    uniform sampler2D texturePosition;
    attribute vec2 particleUv;
    uniform float uDeposit;
    uniform float uAudioLevel;
    uniform float uBeat;
    varying vec2 vParticleUv;
    void main() {
        vec4 posData = texture2D(texturePosition, particleUv);
        vParticleUv = particleUv;
        vec3 pos = vec3((posData.xy - 0.5) * 2.0, 0.0);
        float localPulse = sin(particleUv.x * 71.0 + particleUv.y * 43.0);
        gl_PointSize = 1.25 + uDeposit * 5.0 + max(localPulse, 0.0) * (uAudioLevel * 0.35 + uBeat * 1.2);
        gl_Position = vec4(pos, 1.0);
    }
`;

export const particleFragmentShader = `
    uniform float uDeposit;
    uniform float uAudioLevel;
    uniform float uBeat;
    varying vec2 vParticleUv;
    void main() {
        // Depósito visible después del blur 3x3.
        vec2 point = gl_PointCoord - 0.5;
        float soft = 1.0 - smoothstep(0.2, 0.5, length(point));
        float localPulse = 0.5 + 0.5 * sin(vParticleUv.x * 91.0 + vParticleUv.y * 57.0);
        float audioVariation = 1.0 + uAudioLevel * (localPulse - 0.5) * 0.06 + uBeat * 0.12;
        gl_FragColor = vec4(vec3(uDeposit * soft * audioVariation), 1.0);
    }
`;

// ====================================================================
// 3. SHADER DE DIFUSIÓN Y DECAIMIENTO DEL TRAIL MAP
// ====================================================================
export const trailVertexShader = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = vec4(position, 1.0);
    }
`;

export const trailFragmentShader = `
    uniform sampler2D uPrevTrail;
    uniform vec2 uTexelSize;
    uniform float uDecay;
    varying vec2 vUv;

    void main() {
        // Kernel Blur 3x3
        float sum = 0.0;
        sum += texture2D(uPrevTrail, vUv + vec2(-uTexelSize.x, -uTexelSize.y)).r;
        sum += texture2D(uPrevTrail, vUv + vec2(0.0,          -uTexelSize.y)).r;
        sum += texture2D(uPrevTrail, vUv + vec2(uTexelSize.x,  -uTexelSize.y)).r;
        sum += texture2D(uPrevTrail, vUv + vec2(-uTexelSize.x, 0.0)).r;
        sum += texture2D(uPrevTrail, vUv).r;
        sum += texture2D(uPrevTrail, vUv + vec2(uTexelSize.x,  0.0)).r;
        sum += texture2D(uPrevTrail, vUv + vec2(-uTexelSize.x, uTexelSize.y)).r;
        sum += texture2D(uPrevTrail, vUv + vec2(0.0,          uTexelSize.y)).r;
        sum += texture2D(uPrevTrail, vUv + vec2(uTexelSize.x,  uTexelSize.y)).r;

        float diffused = sum / 9.0;
        gl_FragColor = vec4(vec3(min(diffused * uDecay, 1.0)), 1.0);
    }
`;

// ====================================================================
// 4. SHADER DE POSTPROCESAMIENTO Y COLORIMETRÍA
// ====================================================================
export const displayFragmentShader = `
    uniform sampler2D uTrailMap;
    uniform vec2 uResolution;
    uniform float uAudioLevel;
    uniform float uAudioBass;
    uniform float uBrightness;
    uniform float uPulse;
    uniform float uBeat;
    varying vec2 vUv;

    vec3 getPhysarumColor(float density) {
        // Paleta: Fondo Oscuro -> Azul Claro -> Verde Claro -> Blanco Suave
        vec3 bg         = vec3(0.01, 0.03, 0.07);
        vec3 lightBlue  = vec3(0.22, 0.74, 0.97); // #38bdf8
        vec3 lightGreen = vec3(0.29, 0.87, 0.50); // #4ade80
        vec3 softWhite  = vec3(0.91, 0.94, 0.96); // #f1f5f9

        float d = clamp(density, 0.0, 1.0);

        // Mapeo no lineal mediante smoothstep
        vec3 col = mix(bg, lightBlue, smoothstep(0.0, 0.25, d));
        col = mix(col, lightGreen, smoothstep(0.25, 0.65, d));
        col = mix(col, softWhite, smoothstep(0.65, 1.0, d));
        
        return col;
    }

    void main() {
        float density = texture2D(uTrailMap, vUv).r;
        vec2 texel = 1.5 / uResolution;
        float halo = 0.0;
        halo += texture2D(uTrailMap, vUv + vec2(texel.x, 0.0)).r;
        halo += texture2D(uTrailMap, vUv - vec2(texel.x, 0.0)).r;
        halo += texture2D(uTrailMap, vUv + vec2(0.0, texel.y)).r;
        halo += texture2D(uTrailMap, vUv - vec2(0.0, texel.y)).r;
        halo *= 0.25;

        float luminousDensity = clamp((density * uBrightness + halo * 0.8) * (1.0 + uAudioLevel * 0.08), 0.0, 1.0);
        vec3 color = getPhysarumColor(luminousDensity);
        vec3 pulseColor = mix(vec3(0.22, 0.74, 0.97), vec3(0.29, 0.87, 0.50), smoothstep(0.25, 0.75, uAudioBass));
        color += pulseColor * (uAudioLevel * 0.035 + uPulse * 0.025 + uBeat * 0.18);
        color += vec3(0.004, 0.012, 0.016) * uPulse;
        float vignette = 1.0 - smoothstep(0.35, 0.85, distance(vUv, vec2(0.5)));
        color *= mix(0.78, 1.0, vignette);
        gl_FragColor = vec4(color, 1.0);
    }
`;
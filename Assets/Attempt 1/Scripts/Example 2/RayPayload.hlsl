#define PI 3.14
#define DEG_TO_RAD (PI / 180.0)

struct RayPayload
{
    float4 color;
    float3 worldPos;
    float3 worldDir;
    float4 energy;
};

struct RayPayloadShadow
{
    float shadowValue;
};

float RandomValue(inout uint state)
{
    //state *= (state + 2142323) * (state + 69591) * (state + 10193);
    state = state * 757796405 + 2891336453;
    uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967295.0;
}

// seb lague
float RandomValueNormalDistribution(inout uint state)
{
    float theta = 2 * 3.1415926 * RandomValue(state);
    float rho = sqrt(-2 * log(RandomValue(state)));
    return rho * cos(theta);
}

// seb lague
float3 RandomDirection(inout uint state)
{
    float x = RandomValueNormalDistribution(state);
    float y = RandomValueNormalDistribution(state);
    float z = RandomValueNormalDistribution(state);
    return normalize(float3(x, y, z));
}

// seb lague
float3 RandomHemisphereDirection(float3 normal, inout uint rngState)
{
    float3 dir = RandomDirection(rngState);
    return dir * sign(dot(normal, dir));
}


// TODO: Untested
float3 RandomUniformDir(inout uint randState)
{
    float phi = RandomValue(randState) * 360.0 * DEG_TO_RAD;
    float theta = RandomValue(randState) * 180.0 * DEG_TO_RAD;
    return normalize(float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)));
}
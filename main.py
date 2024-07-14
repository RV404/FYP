from GenerationPrediction import getSolarGeneration,getWindGeneration
from DemandPrediction import getDemandForecast
demand =getDemandForecast()
solar = getSolarGeneration()
wind = getWindGeneration()

print(demand)
print(solar)
print(wind)
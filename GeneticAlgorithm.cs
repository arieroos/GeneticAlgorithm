using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace GeneticAlgorithm
{
    public class Individual<T> where T : ICloneable, IEquatable<T>
    {
        public delegate double Evaluation(Individual<T> individual);

        public List<T> Genome { get; set; }

        public Evaluation Evaluate { get; private set; }
        private double? _fitness { get; set; }

        public Individual(List<T> genome, Evaluation eval)
        {
            Genome = genome ?? new List<T>();
            Evaluate = eval;
            _fitness = null;
        }

        public double Fitness()
        {
            var fitness = _fitness ?? Evaluate(this);
            _fitness = fitness;
            return fitness;
        }
    }


    public class Algorithm<T> where T : ICloneable, IEquatable<T>
    {
        public delegate Individual<T> Mutate(Individual<T> individual, double rate);

        private Mutate _mutate;
        private Individual<T>[] _population { get; set; }
        private bool _sorted { get; set; }

        public double MaxMutationRate { get; set; }
        public double MutationRate { get; set; }
        public double MutationRateAdj { get; set; }
        public double MinMutationRate { get; set; }
        public int ResetCount { get; set; }
        public int CurrentGeneration { get; private set; }

        public Algorithm(
            Mutate mutate,
            Individual<T> adam,
            int populationSize = 10,
            double maxMutationRate = 100,
            double mutationAdj = 10,
            double minMutationRate = 10,
            int resetCount = 40,
            double[] rotations = null
        )
        {
            _mutate = mutate;
            MutationRate = maxMutationRate;

            _population = new Individual<T>[populationSize];
            _population[0] = adam;
            for (int i = 1; i < populationSize; i++)
            {
                _population[i] = _mutate(adam, MutationRate / 100.0);
            }

            MaxMutationRate = maxMutationRate;
            MutationRateAdj = mutationAdj;
            ResetCount = resetCount;

            CurrentGeneration = 0;
            _sorted = false;
        }

        public (Individual<T>, Individual<T>) Mate(Individual<T> dad, Individual<T> mom)
        {
            if (dad.Genome.Count != mom.Genome.Count)
            {
                throw new ArgumentException("Two individuals must have the same length genome to be able to mate.");
            }

            var split = dad.Genome.Count > 7 ? dad.Genome.Count / 2 :
                new Random().Next(dad.Genome.Count / 4, dad.Genome.Count / 4 * 3);

            var dadGenes1 = dad.Genome.Take(split);
            var dadGenes2 = dad.Genome.Skip(split);

            var momGenes1 = mom.Genome.Take(split);
            var momGenes2 = mom.Genome.Skip(split);

            var childGenome1 = dadGenes1.Select(g => (T)g.Clone()).ToList();
            foreach (var gene in mom.Genome)
            {
                if (!childGenome1.Contains(gene))
                {
                    childGenome1.Add((T)gene.Clone());
                }
            }

            var childGenome2 = momGenes1.Select(g => (T)g.Clone()).ToList();
            foreach (var gene in dad.Genome)
            {
                if (!childGenome2.Contains(gene))
                {
                    childGenome2.Add((T)gene.Clone());
                }
            }

            return (new Individual<T>(childGenome1, dad.Evaluate), new Individual<T>(childGenome2, dad.Evaluate));
        }

        public void Generation()
        {
            if (!_sorted)
            {
                EvaluateAndSort();
            }

            var newPop = new Individual<T>[_population.Length];
            newPop[0] = _population[0]; // preserve the best fit to the next generation

            for (int i = 1; i < newPop.Length - 1; i += 2)
            {
                var dad = randomWeightedIndividual();
                var mom = randomWeightedIndividual();

                (newPop[i], newPop[i + 1]) = Mate(mom, dad);

                // if one of the parents are the champ, don't mutate
                if (!(dad == newPop[0] ^ mom == newPop[0]))
                {
                    newPop[i] = _mutate(newPop[i], MutationRate / 100.0);
                    newPop[i + 1] = _mutate(newPop[i + 1], MutationRate / 100.0);
                }
            }
            if (newPop[newPop.Length - 1] == null)
            {
                (var child, _) = Mate(randomWeightedIndividual(), randomWeightedIndividual());
                newPop[newPop.Length - 1] = _mutate(child, MutationRate / 100.0);
            }

            _population = newPop;
            _sorted = false;
            EvaluateAndSort();
            CurrentGeneration++;
        }

        public Individual<T> GetCurrentChamp() => _population[0];

        public void EvaluateAndSort()
        {
            Parallel.ForEach(_population, p => p.Fitness());
            Array.Sort(_population, (a, b) => (int)((a.Fitness() - b.Fitness()) * 1000));
            _sorted = true;
        }

        public void Run(int generations = 0, Action<Individual<T>> callback = null)
        {
            callback = callback ?? (d => { });

            EvaluateAndSort();
            var prevChamp = GetCurrentChamp();
            callback(prevChamp);

            var staleCount = 0;
            for (int i = 0; i < generations || generations == 0; i++)
            {
                Generation();

                MutationRate = Math.Max(MutationRate - Math.Ceiling(MutationRate * (MutationRateAdj / 100)), 10);

                var champ = GetCurrentChamp();
                if (champ.Fitness() >= prevChamp.Fitness())
                {
                    if (ResetCount <= ++staleCount)
                    {
                        MutationRate = MaxMutationRate;
                        staleCount = 0;
                    }
                }
                else
                {
                    callback(champ);
                    staleCount = 0;
                }
                prevChamp = champ;

            }

            callback(_population[0]);
        }

        private Individual<T> randomWeightedIndividual()
        {
            var totalIndex = (_population.Length * (_population.Length + 1)) / 2.0;
            var random = new Random().NextDouble();
            var currentIndex = 0;
            for (int i = 0; currentIndex < totalIndex; i++)
            {
                currentIndex += (_population.Length - i);
                if (random < (currentIndex / totalIndex))
                {
                    return _population[i];
                }
            }

            // logically we shouldn't be here
            return _population[0];
        }
    }
}
import csv
import itertools
import os.path

from numpy import mean


class TimeSample:
    def __init__(self, environment, trainer, learner, seed,t_time,e_time):
        self.environment = environment
        self.trainer = trainer
        self.learner = learner
        self.seed = seed
        self.t_time = t_time
        self.e_time = e_time

    @staticmethod
    def datapoint(environment,trainer,learner,seed):
        trainer_path = os.path.join('logs', environment, trainer)
        learner_dirs = os.listdir(trainer_path)
        learner_dirs = [p for p in learner_dirs if p.startswith(learner)]
        if len(learner_dirs) == 0:
            return TimeSample(environment,trainer,learner,seed,-1,-1)
        learner_path = os.path.join(trainer_path,learner_dirs[0])
        csv_path = os.path.join(learner_path,f'seed-{seed}','times.csv')
        with open(csv_path,'r') as f:
            line = f.readline()
            _,t_time,e_time = line.split(',')
            t_time,e_time = float(t_time), float(e_time)
            return TimeSample(environment, trainer, learner, seed, t_time, e_time)

    def __str__(self):
        return str({k:v for k,v in self.__dict__.items() if isinstance(v,str) or v>=0})


def anova():
    import scipy.stats as stats

    # Supponiamo di avere le misure in liste per ciascun processo
    p1 = [m1, m2, ..., m20]
    p2 = [m1, m2, ..., m20]
    # Continua per tutti i processi

    # Esegui l'ANOVA
    f_val, p_val = stats.f_oneway(p1, p2, ..., pn)

    print(f"F-value: {f_val}")
    print(f"P-value: {p_val}")

    # Interpreta i risultati
    if p_val < 0.05:
        print("C'è una differenza significativa tra le medie dei processi.")
    else:
        print("Non c'è una differenza significativa tra le medie dei processi.")




def main():
    environments = [
        'point_mass',
        'point_mass_2d'
    ]
    trainers = [
        'default',
        'random',
        'alp_gmm',
        'goal_gan',
        'self_paced',
        'self_paced_v2'
    ]
    learners = [
        'ppo',
        'trpo',
        'sac'
    ]
    seeds = range(1,21)

    samples = [TimeSample.datapoint(e,t,l,s) for (e,t,l,s) in itertools.product(environments,trainers,learners,seeds)]
    samples = [s for s in samples if s.t_time>=0]
    print(len(samples),samples[0])

    averages = []
    for e,t,l in itertools.product(environments,trainers,learners):
        filtered_samples = [s for s in samples if s.environment==e and s.trainer==t and s.learner==l]
        t_times = [s.t_time for s in filtered_samples]
        e_times = [s.e_time for s in filtered_samples]
        averages.append(TimeSample(e,t,l,-1,mean(t_times),mean(e_times)))

    print(len(averages),averages[0])

    for t,l in itertools.product(trainers,learners):
        for e in environments:
            s = [s for s in averages if s.environment==e and s.trainer==t and s.learner==l][0]
            print(t,l,e,s.t_time,s.e_time)

    for t in trainers:
        s = '\multirow{3}{*}{'+t+'}'
        for i,l in enumerate(learners):
            sl = '\t'*(8 if i>0 else 1)
            sl += '& ' + l
            samples = [sample for sample in averages if sample.trainer==t and sample.learner==l]
            samples.sort(key=lambda sample: environments.index(sample.environment))
            for sample in samples:
                sl += ' & {:.2f} & {:.2f}'.format(sample.t_time,sample.e_time)

            s+=sl+'\\\\\n'
        print(s)




if __name__ == '__main__':
    main()
#[macro_use(clip)]
extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::{actor_critic::A2C, td::SARSA},
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::Domain,
    fa::{basis::{Composable, fixed::Fourier}, LFA},
    geometry::Space,
    logging,
    policies::parameterised::Gibbs,
};

use rsrl::clip;

use rsrl::consts::{FOUR_THIRDS, G, TWELVE_DEGREES};
use rsrl::geometry::{
    continuous::Interval,
    discrete::Ordinal,
    product::LinearSpace,
    Vector,
};
use ndarray::{Ix1, NdIndex};
use rsrl::domains::{Observation, Transition};
use slog::KV;

const TAU: f64 = 0.02;

const CART_MASS: f64 = 1.0;
const CART_FORCE: f64 = 10.0;

const POLE_COM: f64 = 0.5;
const POLE_MASS: f64 = 0.1;
const POLE_MOMENT: f64 = POLE_COM * POLE_MASS;

const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;

const LIMITS_X: (f64, f64) = (-2.4, 2.4);
const LIMITS_DX: (f64, f64) = (-6.0, 6.0);
const LIMITS_THETA: (f64, f64) = (-TWELVE_DEGREES, TWELVE_DEGREES);
const LIMITS_DTHETA: (f64, f64) = (-2.0, 2.0);

const REWARD_STEP: f64 = 0.0;
const REWARD_TERMINAL: f64 = -1.0;

const ALL_ACTIONS: [f64; 2] = [-1.0 * CART_FORCE, 1.0 * CART_FORCE];

fn runge_kutta4(fx: &Fn(f64, Vector) -> Vector, x: f64, y: Vector, dx: f64) -> Vector {
    let k1 = dx * fx(x, y.clone());
    let k2 = dx * fx(x + dx / 2.0, y.clone() + k1.clone() / 2.0);
    let k3 = dx * fx(x + dx / 2.0, y.clone() + k2.clone() / 2.0);
    let k4 = dx * fx(x + dx, y.clone() + k3.clone());

    y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
}

#[derive(Debug, Clone, Copy)]
enum StateIndex {
    X = 0,
    DX = 1,
    THETA = 2,
    DTHETA = 3,
}

unsafe impl NdIndex<Ix1> for StateIndex {
    #[inline]
    fn index_checked(&self, dim: &Ix1, strides: &Ix1) -> Option<isize> {
        (*self as usize).index_checked(dim, strides)
    }

    #[inline(always)]
    fn index_unchecked(&self, strides: &Ix1) -> isize { (*self as usize).index_unchecked(strides) }
}

#[derive(Debug)]
pub struct CartPole {
    state: Vector,
}

impl CartPole {
    fn new(x: f64, dx: f64, theta: f64, dtheta: f64) -> CartPole {
        CartPole {
            state: Vector::from_vec(vec![x, dx, theta, dtheta]),
        }
    }

    fn update_state(&mut self, a: usize) {
        let fx = |_x, y| CartPole::grad(ALL_ACTIONS[a], &y);
        let mut ns = runge_kutta4(&fx, 0.0, self.state.clone(), TAU);

        ns[StateIndex::X] = clip!(LIMITS_X.0, ns[StateIndex::X], LIMITS_X.1);
        ns[StateIndex::DX] = clip!(LIMITS_DX.0, ns[StateIndex::DX], LIMITS_DX.1);

        ns[StateIndex::THETA] = clip!(LIMITS_THETA.0, ns[StateIndex::THETA], LIMITS_THETA.1);
        ns[StateIndex::DTHETA] = clip!(LIMITS_DTHETA.0, ns[StateIndex::DTHETA], LIMITS_DTHETA.1);

        self.state = ns;
    }

    fn grad(force: f64, state: &Vector) -> Vector {
        let dx = state[StateIndex::DX];
        let theta = state[StateIndex::THETA];
        let dtheta = state[StateIndex::DTHETA];

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let z = (force + POLE_MOMENT * dtheta * dtheta * sin_theta) / TOTAL_MASS;

        let numer = G * sin_theta - cos_theta * z;
        let denom = FOUR_THIRDS * POLE_COM - POLE_MOMENT * cos_theta * cos_theta;

        let ddtheta = numer / denom;
        let ddx = z - POLE_COM * ddtheta * cos_theta;

        Vector::from_vec(vec![dx, ddx, dtheta, ddtheta])
    }
}

impl Default for CartPole {
    fn default() -> CartPole { CartPole::new(0.0, 0.0, 0.0, 0.0) }
}

impl Domain for CartPole {
    type StateSpace = LinearSpace<Interval>;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<Vector<f64>> {
        if self.is_terminal() {
            Observation::Terminal(self.state.clone())
        } else {
            Observation::Full(self.state.clone())
        }
    }

    fn step(&mut self, action: usize) -> Transition<Vector<f64>, usize> {
        let from = self.emit();
        // println!("{:?}", from);

        self.update_state(action);
        let to = self.emit();
        // println!("{:?}", to);
        let reward = self.reward(&from, &to);

        Transition {
            from,
            action,
            reward,
            to,
        }
    }

    fn is_terminal(&self) -> bool {
        let x = self.state[StateIndex::X];
        let theta = self.state[StateIndex::THETA];

        x <= LIMITS_X.0 || x >= LIMITS_X.1 || theta <= LIMITS_THETA.0 || theta >= LIMITS_THETA.1
    }

    fn reward(&self, _: &Observation<Vector<f64>>, to: &Observation<Vector<f64>>) -> f64 {
        match *to {
            Observation::Terminal(_) => REWARD_TERMINAL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        LinearSpace::empty() + Interval::bounded(LIMITS_X.0, LIMITS_X.1)
            + Interval::bounded(LIMITS_DX.0, LIMITS_DX.1)
            + Interval::bounded(LIMITS_THETA.0, LIMITS_THETA.1)
            + Interval::bounded(LIMITS_DTHETA.0, LIMITS_DTHETA.1)
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(2) }
}

fn main() {
    let domain = CartPole::default();

    let n_actions = domain.action_space().card().into();
    let bases = Fourier::from_space(3, domain.state_space()).with_constant();

    let policy = make_shared({
        let fa = LFA::vector(bases.clone(), n_actions);

        Gibbs::new(fa)
    });
    let critic = {
        let q_func = LFA::vector(bases, n_actions);

        SARSA::new(q_func, policy.clone(), 0.1, 0.99)
    };

    let mut agent = A2C::new(critic, policy, 0.01);

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(CartPole::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 10);

        // Realise 1000 episodes of the experiment generator.
        run(e, 10, Some(logger.clone()))
    };
    // println!("{:?}", _training_result[0].steps);

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
    println!("{:?}", ALL_ACTIONS);

}
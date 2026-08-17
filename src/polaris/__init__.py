"""polaris -- a production ML platform for B2B SaaS churn.

Vertex Systems is an invented company and every record is synthetic.

The project is deliberately not a model. A model is an afternoon; the things
that make one usable are the rest of it:

* features computed **as of** a reference date, so nothing the model sees was
  unknowable at prediction time;
* a decision threshold chosen by expected value, because the business acts on
  the prediction and the action has a cost;
* a promotion gate that refuses a challenger which is better on average and
  worse on a segment;
* drift monitoring, because the labels arrive sixty days after the prediction
  and something has to notice in the meantime.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]

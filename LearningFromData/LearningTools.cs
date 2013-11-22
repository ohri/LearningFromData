using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using System.IO;
using libsvm;

namespace LearningFromData
{
    public class Point
    {
        public double x;
        public double y;
        public int fx;

        public Point( double _x, double _y )
        {
            fx = 0;
            x = _x;
            y = _y;
        }

        public DenseVector AsDenseVector()
        {
            DenseVector foo = new DenseVector( 3 );
            foo[0] = 1.0;
            foo[1] = x;
            foo[2] = y;
            return foo;
        }

        public static Point GetRandomPoint( Random r )
        {
            return new Point( r.NextDouble() * 2.0 - 1.0, r.NextDouble() * 2.0 - 1.0 );
        }

        public static Point[] CreatePointSet( int n, Line l, Random r )
        {
            Point[] point_set = new Point[n];
            for( int i = 0; i < n; i++ )
            {
                point_set[i] = GetRandomPoint( r );
                point_set[i].fx = l.CalculateY( point_set[i] );
            }
            return point_set;
        }

        public static Point[] ImportPointSet( string url )
        {
            List<Point> point_set = new List<Point>();

            string line;
            using( StreamReader file = new StreamReader( url ) )
            {
                // if this was real programming, we'd put some error handling around this
                while( ( line = file.ReadLine() ) != null )
                {
                    string[] parts = line.Split( new char[0], StringSplitOptions.RemoveEmptyEntries );
                    Point p = new Point( double.Parse( parts[0] ), double.Parse( parts[1] ) );
                    if( double.Parse( parts[2] ) > 0 )
                    {
                        p.fx = 1;
                    }
                    else
                    {
                        p.fx = -1;
                    }
                    point_set.Add( p );
                }
                file.Close();
            }

            return point_set.ToArray<Point>();
        }
    }

    public struct RunResult
    {
        public int epochs;
        public double eout;

        public RunResult( double _eout, int _epochs )
        {
            epochs = _epochs;
            eout = _eout;
        }
    }

    public class Line
    {
        public Point p1;
        public Point p2;
        public double m;
        public double b;

        public Line( Point _p1, Point _p2 )
        {
            p1 = _p1;
            p2 = _p2;

            m = ( p2.y - p1.y ) / ( p2.x - p1.x );
            b = p1.y - ( m * p1.x );
        }

        public int CalculateY( Point p )
        {
            // compare the y for this new point to f(x) at that point
            if( p.y > ( p.x * m + b ) )
            {
                return 1;
            }
            return -1;
        }

        public static Line CreateRandomLine( Random r )
        {
            return new Line( Point.GetRandomPoint( r ), Point.GetRandomPoint( r ) );
        }
    }

    public class LearningTools
    {

        public static List<int> GeneratePermutations( int n, Random r )
        {
            // put all n numbers in a list
            List<int> foo = new List<int>();
            for( int i = 0; i < n; i++ )
            {
                foo.Add( i );
            }

            // now pull them out randomly and add them to the return list
            List<int> new_perm = new List<int>();
            for( int i = 0; i < n; i++ )
            {
                int picked = r.Next( foo.Count );
                new_perm.Add( foo[picked] );
                foo.RemoveAt( picked );
            }

            return new_perm;
        }

        // this is a VERY specific logistic regression gradient
        // need to generalize this
        // maybe logistic regression machine as an abstract base, requires calclrgradient 
        // in specific implementations?
        static DenseVector CalculateLRGradient( DenseVector w, Point p )
        {
            // xi starts as the vector for x, then gets transformed into the gradient
            DenseVector xi = p.AsDenseVector();

            double denom = 1.0 + Math.Pow( Math.E, (double)p.fx * xi.DotProduct( w ) );

            xi.Multiply( -1.0 * (double)p.fx, xi );
            xi.Divide( denom, xi );

            return xi;
        }

        public static RunResult RunLogisticRegression( int num_points, int num_testing_points, double learning_rate, double stop_when, Random r )
        {
            // create a line from two random points
            Line l = Line.CreateRandomLine( r );

            //const int NUM_POINTS = 100;
            //const int NUM_TESTING_POINTS = 100;
            //const double LEARNING_RATE = 0.01;
            //const double STOP_WHEN = 0.01;

            // create the training points
            Point[] training_points = Point.CreatePointSet( num_points, l, r );

            DenseVector w = new DenseVector( 3 );
            DenseVector w_new = new DenseVector( 3 );

            int epochs = 0;
            double norm = 0;
            do
            {
                // copy over the weight from the last epoch
                w[0] = w_new[0];
                w[1] = w_new[1];
                w[2] = w_new[2];

                // need to define a permutation of the 100 data points
                // then train with them one by one
                List<int> permutation = GeneratePermutations( num_points, r );
                for( int i = 0; i < num_points; i++ )
                {
                    // effectively, w(t+1) = w(0) - eta * gradient
                    DenseVector gradient = CalculateLRGradient( w_new, training_points[permutation[i]] );
                    w_new[0] -= learning_rate * gradient[0];
                    w_new[1] -= learning_rate * gradient[1];
                    w_new[2] -= learning_rate * gradient[2];
                }

                epochs++;
                norm = Math.Sqrt( Math.Pow( w[0] - w_new[0], 2 ) + Math.Pow( w[1] - w_new[1], 2 ) + Math.Pow( w[2] - w_new[2], 2 ) );
            }
            while( norm >= stop_when );

            // create testing points
            double eout = 0;
            for( int i = 0; i < num_testing_points; i++ )
            {
                // create a testing point
                Point p = Point.GetRandomPoint( r );
                p.fx = l.CalculateY( p );
                DenseVector xn = p.AsDenseVector();

                // throw it up against h( x ) to calcualte eout (cross entropy error)
                eout += Math.Log( 1.0 + Math.Pow( Math.E, -1.0 * (double)p.fx * xn.DotProduct( w_new ) ) );
            }
            eout /= (double)num_testing_points; // average it

            return new RunResult( eout, epochs );
        }

        public static DenseVector RunLinearRegression( int num_points, DenseMatrix X, DenseVector y, Random r )
        {
            // create the target function
            Line l = Line.CreateRandomLine( r );

            //var y = new DenseVector( num_points );
            //for( int i = 0; i < num_points; i++ )
            //{
            //    y[i] = training_points[i].fx;
            //}

            //var X = new DenseMatrix( num_points, 3 );
            //for( int i = 0; i < num_points; i++ )
            //{
            //    X[i, 0] = 1;
            //    X[i, 1] = training_points[i].x;
            //    X[i, 2] = training_points[i].y;
            //}

            // train with linear regression
            // linear regression gives us a new equation for a line in 'one fell swoop'
            // w = Xdagger * y
            // where w and y are vectors and Xdagger is matrix of inverse( Xtrans*X ) * Xtrans
            var Xdagger = ( ( ( X.Transpose() ).Multiply( X ) ).Inverse() ).Multiply( X.Transpose() );
            return (DenseVector)Xdagger.Multiply( y );
        }

        public static DenseVector RunLinRegRegularized( int num_points, DenseMatrix X, DenseVector y, double lambda, Random r )
        {
            // ( Xt*X + lambda*I )^(-1)( Xt ) * y
            var Xdagger = ( X.TransposeThisAndMultiply( X ).Add( DenseMatrix.Identity( X.ColumnCount ).Multiply( lambda ) ).Inverse() ).Multiply( X.Transpose() );
            return (DenseVector)Xdagger.Multiply( y );
        }

        public static int RunPerceptron( int num_points, Point[] training_points, DenseVector w, Random r )
        {
            int iterations = 0;
            bool some_wrong = true;
            while( some_wrong )
            {
                iterations++;
                // get the set of misclassified points
                List<Point> misclassified = new List<Point>();
                foreach( Point p in training_points )
                {
                    double PointY = (double)w[0] + (double)w[1] * p.x + (double)w[2] * p.y;
                    int sign = -1;
                    if( PointY > 0.0 )
                    {
                        sign = 1;
                    }
                    if( sign != p.fx )
                    {
                        misclassified.Add( p );
                    }
                }

                if( misclassified.Count() == 0 )
                {
                    some_wrong = false;
                    break;
                }

                // randomly pick one
                int bad_point_index = r.Next( 0, misclassified.Count() );

                // update w
                w[0] += (double)( misclassified[bad_point_index].fx );
                w[1] += (double)( misclassified[bad_point_index].fx ) * misclassified[bad_point_index].x;
                w[2] += (double)( misclassified[bad_point_index].fx ) * misclassified[bad_point_index].y;
            }

            return iterations;
        }

        public static int TestPoints( DenseVector w, Point[] test_points )
        {
            int fails = 0;
            foreach( Point p in test_points )
            {
                int sign = -1;
                if( (double)w[0] + (double)w[1] * p.x + (double)w[2] * p.y > 0.0 )
                {
                    sign = 1;
                }
                if( sign != p.fx )
                {
                    fails++;
                }
            }
            return fails;
        }
    }

    public class SVMHelper
    {
        public svm _svm;
        public svm_problem _prob;
        public svm_parameter _param;
        public svm_model _model;

        public SVMHelper( Point[] x )
        {
            _prob = new svm_problem();
            _svm = new svm();
            _param = new svm_parameter();

            // these are defaults copied from the github java version
            _param.svm_type = svm_parameter.C_SVC;
            _param.kernel_type = svm_parameter.LINEAR;
            _param.degree = 3;
            _param.coef0 = 0;
            _param.nu = 0.5;
            _param.cache_size = 100;
            _param.C = 10000;
            _param.eps = .001;
            _param.p = 0.1;
            _param.shrinking = 1;
            _param.probability = 0;
            _param.nr_weight = 0;
            _param.weight_label = new int[0];
            _param.weight = new double[0];
            _param.gamma = 0;

            // put in the y's
            _prob.y = new double[x.Count()];
            for( int j = 0; j < x.Count(); j++ )
            {
                _prob.y[j] = x[j].fx;
            }

            _prob.l = _prob.y.Count();

            // put in the x's
            _prob.x = new svm_node[x.Count()][];
            for( int j = 0; j < x.Count(); j++ )
            {
                _prob.x[j] = new svm_node[2];
                _prob.x[j][0] = new svm_node();
                _prob.x[j][0].index = 0;
                _prob.x[j][0].value_Renamed = x[j].x;
                _prob.x[j][1] = new svm_node();
                _prob.x[j][1].index = 1;
                _prob.x[j][1].value_Renamed = x[j].y;
            }

        }

        public void train()
        {
            _model = svm.svm_train( _prob, _param );
        }
    }
}

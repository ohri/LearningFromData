﻿using System;
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

        public static DenseVector RunRegularRBF( int n, int k, Data[] training_points, Data[] mu, double gamma, double[] y, Random r )
        {
            DenseMatrix phi = new DenseMatrix( n, k + 1 );
            for( int h = 0; h < n; h++ )
            {
                phi[h, 0] = 1.0;
                for( int g = 0; g < k; g++ )
                {
                    phi[ h, g + 1 ] = Math.Exp( -1.0 * gamma * Math.Pow( training_points[h].AsDenseVector().Subtract( mu[g].AsDenseVector() ).Norm( 2 ), 2 ) );
                }
            }
            return LearningTools.RunLinearRegression( n, phi, new DenseVector( y ), r );
        }

        public static int RunLloydsAlgorithm( Data[] training_points, out Data[] mu, int k, int d, Random r )
        {
            bool converged = false;
            bool bad_run = false;
            int convergence_count = 0;

            // pick k random centers for clusters
            mu = Data.CreateDataSet( k, null, r, d );

            while( !converged && !bad_run )
            {
                convergence_count++;

                // assign each data point to a cluster
                List<int>[] clusters = new List<int>[k];
                for( int j = 0; j < k; j++ )
                {
                    clusters[j] = new List<int>();
                }
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    double[] distance = new double[k];
                    for( int m = 0; m < k; m++ )
                    {
                        distance[m] = training_points[j].PointDistance( mu[m] );
                    }
                    double min = distance[0];
                    int index = 0;
                    for( int m = 0; m < k; m++ )
                    {
                        if( distance[m] < min )
                        {
                            index = m;
                            min = distance[m];
                        }
                    }
                    clusters[index].Add( j );
                }

                // make sure there are no empty clusters
                for( int j = 0; j < k; j++ )
                {
                    if( clusters[j].Count() == 0 )
                    {
                        bad_run = true;
                    }
                }

                // create new mu's
                Data[] new_mu = new Data[k];
                for( int j = 0; j < k; j++ )
                {
                    // average the points in the cluster - this becomes the new center
                    new_mu[j] = new Data( new double[d] );
                    for( int m = 0; m < clusters[j].Count(); m++ )
                    {
                        for( int mi = 0; mi < d; mi++ )
                        {
                            new_mu[j].x[mi] += training_points[clusters[j][m]].x[mi];
                        }
                    }
                    for( int m = 0; m < d; m++ )
                    {
                        new_mu[j].x[m] = new_mu[j].x[m] / (double)clusters[j].Count();
                    }
                }

                // have we converged?
                bool difference = false;
                for( int j = 0; j < k; j++ )
                {
                    if( !mu[j].Equals( new_mu[j] ) )
                    {
                        difference = true;
                    }
                }
                if( !difference )
                {
                    converged = true;
                }
                else
                {
                    mu = new_mu;
                }
            }

            if( converged )
            {
                return convergence_count;
            }
            else
            {
                return -1;
            }
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

        public static int RunPerceptron( int num_points, Data[] training_data, DenseVector w, Random r )
        {
            Point[] training_points = Data.AsPoints( training_data );

            return RunPerceptron( num_points, training_points, w, r );
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
                    double PointY = w[0] + w[1] * p.x + w[2] * p.y;
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

        public static int TestPoints( DenseVector w, Data[] test_data )
        {
            Point[] test_points = Data.AsPoints( test_data );

            return TestPoints( w, test_points );
        }

        public static int TestPoints( DenseVector w, Point[] test_points )
        {
            int fails = 0;
            foreach( Point p in test_points )
            {
                int sign = -1;
                if( w[0] + w[1] * p.x + w[2] * p.y > 0.0 )
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
        public svm svm;
        public svm_problem prob;
        public svm_parameter param;
        public svm_model model;
        public DenseVector w;
        public double b;

        public SVMHelper( Data[] x )
        {
            prob = new svm_problem();
            svm = new svm();
            param = new svm_parameter();

            // these are defaults copied from the github java version
            param.svm_type = svm_parameter.C_SVC;
            param.kernel_type = svm_parameter.LINEAR;
            param.degree = 3;
            param.coef0 = 0;
            param.nu = 0.5;
            param.cache_size = 100;
            param.C = 10000;
            param.eps = .001;
            param.p = 0.1;
            param.shrinking = 1;
            param.probability = 0;
            param.nr_weight = 0;
            param.weight_label = new int[0];
            param.weight = new double[0];
            param.gamma = 0.5;

            // put in the y's
            prob.y = new double[x.Count()];
            for( int j = 0; j < x.Count(); j++ )
            {
                prob.y[j] = x[j].y;
            }

            prob.l = prob.y.Count();

            // put in the x's
            prob.x = new svm_node[x.Count()][];
            for( int j = 0; j < x.Count(); j++ )
            {
                prob.x[j] = new svm_node[x[j].x.Count()];
                for( int k = 0; k < x[0].x.Count(); k++ )
                {
                    prob.x[j][k] = new svm_node();
                    prob.x[j][k].index = k;
                    prob.x[j][k].value_Renamed = x[j].x[k];
                }
            }
        }

        public void train()
        {
            model = svm.svm_train( prob, param );

            // w = model.SV * model.sv_coef
            // b = -model.rho
            // if model.Label(1) == -1, multiply w and b by -1

            // calculate w
            w = new DenseVector( model.SV[0].Count() );
            for( int i = 0; i < model.SV.Count(); i++ )
            {
                for( int j = 0; j < model.SV[i].Count(); j++ )
                {
                    w[j] += model.SV[i][j].value_Renamed * model.sv_coef[0][i];
                }
            }

            b = model.rho[0] * -1.0;

            // this is right out of their faq (as are the above descriptions)
            // i believe the labels are the classifications (-1, 1), so it would appear
            // to ALWAYS need to multiply these for the standard 1, -1 classification problem
            if( model.label[0] == -1 )
            {
                w.Multiply( -1.0 );
                b *= -1.0;
            }
        }

        // returns Ecv
        public double cross_validation( int k )
        {
            int total_wrong = 0;
            double[] target = new double[prob.l];
            svm.svm_cross_validation( prob, param, k, target );
            for( int i = 0; i < prob.l; i++ )
            {
                if( target[i] != prob.y[i] )
                {
                    total_wrong++;
                }
            }
            return (double)total_wrong / (double)prob.l;
        }

        public double predict( Data[] x )
        {
            int fail = 0;

            for( int i = 0; i < x.Count(); i++ )
            {
                svm_node[] p = new svm_node[2];
                p[0] = new svm_node();
                p[0].index = 0;
                p[0].value_Renamed = x[i].x[0];
                p[1] = new svm_node();
                p[1].index = 1;
                p[1].value_Renamed = x[i].x[1];

                double ysvm = svm.svm_predict( model, p );
                if( Math.Sign( ysvm ) != x[i].y )
                {
                    fail++;
                }
            }

            return (double)fail / (double)x.Count();
        }
    }

    public class Data
    {
        public double[] x;
        public int y;

        public Data( double[] _x )
        {
            x = _x;
            y = 0;
        }

        // NOTE: does not compare y!
        public bool Equals( Data b )
        {
            if( x.Count() != b.x.Count() )
            {
                return false;
            }

            for( int i = 0; i < x.Count(); i++ )
            {
                if( x[i] != b.x[i] )
                {
                    return false;
                }
            }

            return true;
        }

        public double PointDistance( Data b )
        {
            if( x.Count() != b.x.Count() )
            {
                throw new Exception();
            }

            double sum = 0;
            for( int i = 0; i < x.Count(); i++ )
            {
                sum += Math.Pow( x[i] - b.x[i], 2 );
            }

            return Math.Sqrt( sum );
        }
        
        public static Data[] CreateDataSet( int n, Line l, Random r, int d )
        {
            Data[] point_set = new Data[n];
            for( int i = 0; i < n; i++ )
            {
                point_set[i] = GetRandomData( d, r );

                // this is a terrible bastardization for the 2D case
                // when i finally get a lambda or something in there, i 
                // can improve this
                if( l != null )
                {
                    point_set[i].y = l.CalculateY( point_set[i].AsPoint() );
                }
            }
            return point_set;
        }

        public DenseVector AsDenseVector()
        {
            double[] copy_of_x = new double[x.Count()];
            x.CopyTo( copy_of_x, 0 );
            return new DenseVector( copy_of_x );
        }

        public static Data GetRandomData( int d, Random r )
        {
            double[] x = new double[d];
            for( int i = 0; i < d; i++ )
            {
                x[i] = r.NextDouble() * 2.0 - 1.0;
            }
            return new Data( x );
        }

        public Point AsPoint()
        {
            Point p = new Point( x[0], x[1] );
            p.fx = y;
            return p;
        }

        public static Point[] AsPoints( Data[] training_data )
        {
            Point[] training_points = new Point[training_data.Count()];
            for( int i = 0; i < training_data.Count(); i++ )
            {
                training_points[i] = training_data[i].AsPoint();
            }
            return training_points;
        }
    }
}

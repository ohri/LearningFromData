using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningFromData;
using MathNet.Numerics.LinearAlgebra.Double;

namespace LearningFromData
{
    class W6
    {
        static void Main( string[] args )
        {
            W6P2();
            W6P3();
        }

        static void W6P2()
        {
            Point[] training_points = Point.ImportPointSet( @"c:\LFD_W6_P2_insample.txt" );
            Point[] out_points = Point.ImportPointSet( @"c:\LFD_W6_P2_outsample.txt" );

            // transform these to Z
            DenseMatrix X = new DenseMatrix( training_points.Count(), 8 );
            DenseVector y = new DenseVector( training_points.Count() );
            for( int i = 0; i < training_points.Count(); i++ )
            {
                X[i, 0] = 1;
                X[i, 1] = training_points[i].x;
                X[i, 2] = training_points[i].y;
                X[i, 3] = Math.Pow( training_points[i].x, 2 );
                X[i, 4] = Math.Pow( training_points[i].y, 2 );
                X[i, 5] = training_points[i].x * training_points[i].y;
                X[i, 6] = Math.Abs( training_points[i].x - training_points[i].y );
                X[i, 7] = Math.Abs( training_points[i].x + training_points[i].y );

                y[i] = training_points[i].fx;
            }

            Random r = new Random();
            DenseVector w = LearningTools.RunLinearRegression( training_points.Count(), X, y, r );

            // now calculate in sample error
            int num_wrong = 0;
            for( int i = 0; i < training_points.Count(); i++ )
            {
                if( Math.Sign( X.Row( i ).DotProduct( w ) ) != training_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double ein = (double)num_wrong / (double)training_points.Count();

            // and out of sample error
            num_wrong = 0;
            for( int i = 0; i < out_points.Count(); i++ )
            {
                DenseVector x = new DenseVector( 8 );
                x[0] = 1;
                x[1] = out_points[i].x;
                x[2] = out_points[i].y;
                x[3] = Math.Pow( out_points[i].x, 2 );
                x[4] = Math.Pow( out_points[i].y, 2 );
                x[5] = out_points[i].x * out_points[i].y;
                x[6] = Math.Abs( out_points[i].x - out_points[i].y );
                x[7] = Math.Abs( out_points[i].x + out_points[i].y );

                if( Math.Sign( x.DotProduct( w ) ) != out_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double eout = (double)num_wrong / (double)out_points.Count();

            Console.WriteLine( "Ein is " + ein.ToString() + " and Eout is " + eout.ToString() );
            string dc = Console.ReadLine();
        }

        static void W6P3()
        {
            Point[] training_points = Point.ImportPointSet( @"c:\LFD_W6_P2_insample.txt" );
            Point[] out_points = Point.ImportPointSet( @"c:\LFD_W6_P2_outsample.txt" );

            // transform these to Z
            DenseMatrix X = new DenseMatrix( training_points.Count(), 8 );
            DenseVector y = new DenseVector( training_points.Count() );
            for( int i = 0; i < training_points.Count(); i++ )
            {
                X[i, 0] = 1;
                X[i, 1] = training_points[i].x;
                X[i, 2] = training_points[i].y;
                X[i, 3] = Math.Pow( training_points[i].x, 2 );
                X[i, 4] = Math.Pow( training_points[i].y, 2 );
                X[i, 5] = training_points[i].x * training_points[i].y;
                X[i, 6] = Math.Abs( training_points[i].x - training_points[i].y );
                X[i, 7] = Math.Abs( training_points[i].x + training_points[i].y );

                y[i] = training_points[i].fx;
            }

            int k = 2;
            double lambda = Math.Pow( 10, k );

            Random r = new Random();
            DenseVector w = LearningTools.RunLinRegRegularized( training_points.Count(), X, y, lambda, r );

            // now calculate in sample error
            int num_wrong = 0;
            for( int i = 0; i < training_points.Count(); i++ )
            {
                if( Math.Sign( X.Row( i ).DotProduct( w ) ) != training_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double ein = (double)num_wrong / (double)training_points.Count();

            // and out of sample error
            num_wrong = 0;
            for( int i = 0; i < out_points.Count(); i++ )
            {
                DenseVector x = new DenseVector( 8 );
                x[0] = 1;
                x[1] = out_points[i].x;
                x[2] = out_points[i].y;
                x[3] = Math.Pow( out_points[i].x, 2 );
                x[4] = Math.Pow( out_points[i].y, 2 );
                x[5] = out_points[i].x * out_points[i].y;
                x[6] = Math.Abs( out_points[i].x - out_points[i].y );
                x[7] = Math.Abs( out_points[i].x + out_points[i].y );

                if( Math.Sign( x.DotProduct( w ) ) != out_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double eout = (double)num_wrong / (double)out_points.Count();

            Console.WriteLine( "With regularization, Ein is " + ein.ToString() + " and Eout is " + eout.ToString() );
            string dc = Console.ReadLine();
        }
    }
}

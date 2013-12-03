using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningFromData;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;

namespace FinalExam
{
    class Program
    {
        static void Main( string[] args )
        {
            Random r = new Random();

            //int[] digits = new int[5] { 5, 6, 7, 8, 9 };
            //double[] ein = new double[digits.Count()];
            //double[] eout = new double[digits.Count()];
            //FEP7( r, digits, ein, eout, true );

            //int[] digits = new int[5] { 0, 1, 2, 3, 4 };
            //double[] ein = new double[digits.Count()];
            //double[] eout = new double[digits.Count()];
            //FEP8( r, digits, ein, eout, true );

            //FEP9();

            //FEP10();

            //FEP12();

            //FEP13();


            FEP14();
        }

        static void GetFE13Points( int n, int d, out Data[] x, out double[] y, Random r )
        {
            x = Data.CreateDataSet( n, null, r, d );
            y = new double[x.Count()];
            for( int j = 0; j < n; j++ )
            {
                x[j].y = Math.Sign( x[j].x[1] - x[j].x[0] + 0.25 * Math.Sin( Math.PI * x[j].x[0] ) );
                y[j] = x[j].y;
            }
        }

        static void FEP14()
        {
            int d = 2;
            int n = 100;
            int runs = 1000;
            int k = 9;
            double gamma = 1.5;
            int num_test_points = 1000;

            Random r = new Random();

            int svm_wins = 0;
            int rbf_wins = 0;
            int num_bad_runs = 0;

            double avg_svm_eout = 0;
            double avg_rbf_eout = 0;
            double avg_rbf_ein = 0;
            double avg_iterations = 0;

            int perfect_ein = 0;

            for( int i = 0; i < runs; i++ )
            {
                // generate 100 points
                Data[] training_points = new Data[n];
                double[] y = new double[n];
                GetFE13Points( n, d, out training_points, out y, r );

                // train it
                SVMHelper svm = new SVMHelper( training_points );
                svm.param.C = 10000;
                svm.param.gamma = gamma;
                svm.param.kernel_type = libsvm.svm_parameter.RBF;
                svm.prob.y = y;
                svm.prob.l = y.Count();
                svm.train();

                // get the rbf with lloyds

                bool converged = false;
                bool bad_run = false;
                int convergence_count = 0;

                // pick k random centers for clusters
                Data[] mu = Data.CreateDataSet( k, null, r, d );

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

                if( !bad_run )
                {
                    avg_iterations += (double)convergence_count;

                    // generate test points
                    Data[] test_points = null;
                    double[] y_test = null;
                    GetFE13Points( num_test_points, d, out test_points, out y_test, r );

                    // get eout for svm.rbf
                    double svm_eout = svm.predict( test_points );

                    // get eout for my rbf
                    DenseMatrix phi = new DenseMatrix( n, k );
                    for( int g = 0; g < k; g++ )
                    {
                        for( int h = 0; h < n; h++ )
                        {
                            phi[h, g] = Math.Exp( -1.0 * gamma * Math.Pow( training_points[h].AsDenseVector().Subtract( mu[g].AsDenseVector() ).Norm( 2 ), 2 ) );
                        }
                    }
                    DenseVector w = LearningTools.RunLinearRegression( n, phi, new DenseVector( y ), r );

                    // Eout
                    int fail = 0;
                    for( int g = 0; g < test_points.Count(); g++ )
                    {
                        double temp = 0;
                        for( int h = 0; h < k; h++ )
                        {
                            temp += w[h] * Math.Exp( -1.0 * gamma * Math.Pow( test_points[g].AsDenseVector().Subtract( mu[h].AsDenseVector() ).Norm( 2 ), 2 ) );
                        }
                        if( Math.Sign( temp ) != test_points[g].y )
                        {
                            fail++;
                        }
                    }
                    double rbf_eout = (double)fail / (double)test_points.Count();

                    // Ein
                    fail = 0;
                    for( int g = 0; g < training_points.Count(); g++ )
                    {
                        double temp = 0;
                        for( int h = 0; h < k; h++ )
                        {
                            temp += w[h] * Math.Exp( -1.0 * gamma * Math.Pow( training_points[g].AsDenseVector().Subtract( mu[h].AsDenseVector() ).Norm( 2 ), 2 ) );
                        }
                        if( Math.Sign( temp ) != training_points[g].y )
                        {
                            fail++;
                        }
                    }
                    double rbf_ein = (double)fail / (double)training_points.Count();
                    if( fail == 0 )
                    {
                        perfect_ein++;
                    }

                    // declare a winner
                    if( svm_eout < rbf_eout )
                    {
                        svm_wins++;
                    }
                    else
                    {
                        rbf_wins++;
                    }

                    avg_rbf_eout += rbf_eout;
                    avg_svm_eout += svm_eout;
                    avg_rbf_ein += rbf_ein;
                }
                else
                {
                    num_bad_runs++;
                }
            }

            avg_svm_eout = avg_svm_eout / (double)( svm_wins + rbf_wins );
            avg_rbf_eout = avg_rbf_eout / (double)( svm_wins + rbf_wins );
            avg_rbf_ein = avg_rbf_ein / (double)( svm_wins + rbf_wins );
            avg_iterations = avg_iterations / (double)( svm_wins + rbf_wins );

            Console.WriteLine();
            Console.WriteLine( "For K = " + k.ToString() + ", gamma = " + gamma.ToString( "f2" ) );
            Console.WriteLine( "Ran " + runs.ToString() + " runs" );
            Console.WriteLine( "  svm.rbf won " + ( (double)svm_wins / (double)( svm_wins + rbf_wins ) * 100.0 ).ToString( "f2" ) + "%" );
            Console.WriteLine( "  my rbf  won " + ( (double)rbf_wins / (double)( svm_wins + rbf_wins ) * 100.0 ).ToString( "f2" ) + "%" );
            Console.WriteLine( "Avg Eout for svm.rbf was " + avg_svm_eout.ToString( "f4" ) );
            Console.WriteLine( "Avg Eout for my  rbf was " + avg_rbf_eout.ToString( "f4" ) );
            Console.WriteLine( "Avg Ein for my  rbf was " + avg_rbf_ein.ToString( "f4" ) );
            Console.WriteLine( "Perfect Ein for my rbf happened " + perfect_ein.ToString() );
            Console.WriteLine( "Avg lloyd's iterations was " + avg_iterations.ToString( "f2" ) );
            Console.ReadLine();
        }

        static void FEP13()
        {
            int d = 2;
            int n = 100;
            int runs = 1000;

            Random r = new Random();

            int not_sep = 0;

            for( int i = 0; i < runs; i++ )
            {
                // generate 100 points
                Data[] x = Data.CreateDataSet( n, null, r, d );
                double[] y = new double[x.Count()];
                for( int j = 0; j < n; j++ )
                {
                    x[j].y = Math.Sign( x[j].x[1] - x[j].x[0] + 0.25 * Math.Sin( Math.PI * x[j].x[0] ) );
                    y[j] = x[j].y;
                }

                // train it
                SVMHelper svm = new SVMHelper( x );
                svm.param.C = 10000;
                svm.param.gamma = 1.5;
                svm.param.kernel_type = libsvm.svm_parameter.RBF;
                svm.prob.y = y;
                svm.prob.l = y.Count();
                svm.train();

                // see if it was lin sep (ein = 0 => lin sep) 
                if( svm.predict( x ) > 0.0 )
                {
                    not_sep++;
                }
            }

            Console.WriteLine();
            Console.WriteLine( "Ran " + runs.ToString() + " runs, failed to seperate " + not_sep.ToString() + " times" );
            Console.ReadLine();
        }

        static void FEP12()
        {
            Data[] x = new Data[7];
            x[0] = new Data( new double[] { 1, 0 } );
            x[0].y = -1;
            x[1] = new Data( new double[] { 0, 1 } );
            x[1].y = -1;
            x[2] = new Data( new double[] { 0, -1 } );
            x[2].y = -1;
            x[3] = new Data( new double[] { -1, 0 } );
            x[3].y = 1;
            x[4] = new Data( new double[] { 0, 2 } );
            x[4].y = 1;
            x[5] = new Data( new double[] { 0, -2 } );
            x[5].y = 1;
            x[6] = new Data( new double[] { -2, 0 } );
            x[6].y = 1;

            double[] y = new double[7] { -1, -1, -1, 1, 1, 1, 1 };

            SVMHelper svm = new SVMHelper( x );
            svm.prob.y = y;
            svm.prob.l = y.Count();

            svm.param.gamma = 1;
            svm.param.kernel_type = libsvm.svm_parameter.POLY;
            svm.param.coef0 = 1;
            svm.param.degree = 2;
            svm.param.C = 100000;
            svm.train();

            Console.WriteLine();
            Console.WriteLine( "Number of support vectors was " + svm.model.SV.Count() );
            Console.ReadLine();
        }

        static void FEP10()
        {
            int d = 2;
            Random r = new Random();
            double[] lambdas = new double[2]{ 0.01, 1.0 };
            double[] ein = new double[lambdas.Count()];
            double[] eout = new double[lambdas.Count()];

            for( int i = 0; i < lambdas.Count(); i++ )
            {
                List<double> y_train = new List<double>();
                List<double[]> raw_data = new List<double[]>();
                DataParseOneVsFive( @"c:\features.train", d, y_train, raw_data );
                Data[] training_points = RawToData( raw_data, y_train );

                List<double> y_test = new List<double>();
                raw_data = new List<double[]>();
                DataParseOneVsFive( @"c:\features.test", d, y_test, raw_data );
                Data[] test_points = RawToData( raw_data, y_test );

                // transform to Z : 1, x1, x2, x1 * x2, x1^2, x2^2
                DenseMatrix Z = new DenseMatrix( training_points.Count(), 6 );
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    Z[j, 0] = 1;
                    Z[j, 1] = training_points[j].x[1];
                    Z[j, 2] = training_points[j].x[2];
                    Z[j, 3] = training_points[j].x[1] * training_points[j].x[2];
                    Z[j, 4] = Math.Pow( training_points[j].x[1], 2 );
                    Z[j, 5] = Math.Pow( training_points[j].x[2], 2 );
                }

                DenseVector w = LearningTools.RunLinRegRegularized( training_points.Count(), Z, new DenseVector( y_train.ToArray() ),
                    lambdas[i], r );

                // calculate Ein
                int num_wrong = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    if( Math.Sign( Z.Row( j ).DotProduct( w ) ) != training_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                ein[i] = (double)num_wrong / (double)training_points.Count();

                // calculate Eout
                num_wrong = 0;
                for( int j = 0; j < test_points.Count(); j++ )
                {
                    DenseVector z = new DenseVector( 6 );
                    z[0] = 1;
                    z[1] = test_points[j].x[1];
                    z[2] = test_points[j].x[2];
                    z[3] = test_points[j].x[1] * test_points[j].x[2];
                    z[4] = Math.Pow( test_points[j].x[1], 2 );
                    z[5] = Math.Pow( test_points[j].x[2], 2 );
                    if( Math.Sign( z.DotProduct( w ) ) != test_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                eout[i] = (double)num_wrong / (double)test_points.Count();
            }

            Console.WriteLine();
            for( int i = 0; i < lambdas.Count(); i++ )
            {
                Console.WriteLine( "For lambda=" + lambdas[i].ToString( "f2" ) + " ein = " + ein[i].ToString( "f04" )
                    + " eout = " + eout[i].ToString( "f04" ) );
            }
            Console.ReadLine();
        }

        static void DataParseOneVsFive( string filename, int d, List<double> y, List<double[]> raw_data )
        {
            string line;
            using( StreamReader file = new StreamReader( filename ) )
            {
                // if this was real programming, we'd put some error handling around this
                while( ( line = file.ReadLine() ) != null )
                {
                    double[] point = new double[d+1];
                    string[] parts = line.Split( new char[0], StringSplitOptions.RemoveEmptyEntries );
                    double yn = double.Parse( parts[0] );
                    point[0] = 1;
                    if( yn == 1 )
                    {
                        y.Add( 1 );
                        point[1] = double.Parse( parts[1] );
                        point[2] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                    else if( yn == 5 )
                    {
                        y.Add( -1 );
                        point[1] = double.Parse( parts[1] );
                        point[2] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                }
                file.Close();
            }
        }

        static void FEP9()
        {
            Random r = new Random();
            int[] digits = new int[10] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            double[] ein_x = new double[digits.Count()];
            double[] eout_x = new double[digits.Count()];
            double[] ein_z = new double[digits.Count()];
            double[] eout_z = new double[digits.Count()];
            FEP7( r, digits, ein_x, eout_x, false );
            FEP8( r, digits, ein_z, eout_z, false );

            //      Ein                     Eout
            // 0  : X 0.1234  Z 0.1234  --  X 0.1234  Z 0.1234

            Console.WriteLine( "      Ein                     Eout" );
            for( int i = 0; i < digits.Count(); i++ )
            {
                Console.WriteLine( digits[i].ToString() 
                    + "  : X "
                    + ein_x[i].ToString( "f4" )
                    + "  Z "
                    + eout_x[i].ToString( "f4" )
                    + "  --  X "
                    + ein_z[i].ToString( "f4" )
                    + "  Z "
                    + eout_z[i].ToString( "f4" ) );
            }

            Console.WriteLine( "digit, ein_x, eout_x, ein_z, eout_z" );
            for( int i = 0; i < digits.Count(); i++ )
            {
                Console.WriteLine( digits[i].ToString()
                    + ","
                    + ein_x[i].ToString( "f6" )
                    + ","
                    + eout_x[i].ToString( "f6" )
                    + ","
                    + ein_z[i].ToString( "f6" )
                    + ","
                    + eout_z[i].ToString( "f6" ) );
            }
            
            
            Console.ReadLine();
        }

        static void FEP8( Random r, int[] digits, double[] ein, double[] eout, bool print )
        {
            int d = 2;

            for( int i = 0; i < digits.Count(); i++ )
            {                
                List<double> y_train = new List<double>();
                List<double[]> raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.train", d, y_train, raw_data, digits[i] );
                Data[] training_points = RawToData( raw_data, y_train );

                List<double> y_test = new List<double>();
                raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.test", d, y_test, raw_data, digits[i] );
                Data[] test_points = RawToData( raw_data, y_test );

                // transform to Z : 1, x1, x2, x1 * x2, x1^2, x2^2
                DenseMatrix Z = new DenseMatrix( training_points.Count(), 6 );
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    Z[j, 0] = 1;
                    Z[j, 1] = training_points[j].x[1];
                    Z[j, 2] = training_points[j].x[2];
                    Z[j, 3] = training_points[j].x[1] * training_points[j].x[2];
                    Z[j, 4] = Math.Pow( training_points[j].x[1], 2 );
                    Z[j, 5] = Math.Pow( training_points[j].x[2], 2 );
                }

                DenseVector w = LearningTools.RunLinRegRegularized( training_points.Count(), Z, new DenseVector( y_train.ToArray() ),
                    1, r );

                // calculate Ein
                int num_wrong = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    if( Math.Sign( Z.Row( j ).DotProduct( w ) ) != training_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                ein[i] = (double)num_wrong / (double)training_points.Count();

                // calculate Eout
                num_wrong = 0;
                for( int j = 0; j < test_points.Count(); j++ )
                {
                    DenseVector z = new DenseVector( 6 );
                    z[0] = 1;
                    z[1] = test_points[j].x[1];
                    z[2] = test_points[j].x[2];
                    z[3] = test_points[j].x[1] * test_points[j].x[2];
                    z[4] = Math.Pow( test_points[j].x[1], 2 );
                    z[5] = Math.Pow( test_points[j].x[2], 2 );
                    if( Math.Sign( z.DotProduct( w ) ) != test_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                eout[i] = (double)num_wrong / (double)test_points.Count();
            }

            if( print )
            {
                Console.WriteLine( "--No Transform--" );
                for( int i = 0; i < digits.Count(); i++ )
                {
                    Console.WriteLine( "For " + digits[i].ToString() + " eout = " + eout[i].ToString( "f04" )
                        + " ein = " + ein[i].ToString( "f04" ) );
                }
                Console.ReadLine();
            }
        }

        static void FEP7( Random r, int[] digits, double[] ein, double[] eout, bool print )
        {
            int d = 2;

            for( int i = 0; i < digits.Count(); i++ )
            {
                List<double> y_train = new List<double>();
                List<double[]> raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.train", d, y_train, raw_data, digits[i] );
                Data[] training_points = RawToData( raw_data, y_train );

                List<double> y_test = new List<double>();
                raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.test", d, y_test, raw_data, digits[i] );
                Data[] test_points = RawToData( raw_data, y_test );

                DenseMatrix X = new DenseMatrix( training_points.Count(), d + 1 );
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    for( int k = 0; k <= d; k++ )
                    {
                        X[j, k] = training_points[j].x[k];
                    }
                }

                DenseVector w = LearningTools.RunLinRegRegularized( training_points.Count(), X, new DenseVector( y_train.ToArray() ),
                    1, r );

                // calculate Ein
                int num_wrong = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    if( Math.Sign( X.Row( j ).DotProduct( w ) ) != training_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                ein[i] = (double)num_wrong / (double)training_points.Count();

                // calculate Eout
                num_wrong = 0;
                for( int j = 0; j < test_points.Count(); j++ )
                {
                    if( Math.Sign( test_points[j].AsDenseVector().DotProduct( w ) ) != test_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                eout[i] = (double)num_wrong / (double)test_points.Count();
            }

            if( print )
            {
                Console.WriteLine( "--With Transform--" );
                for( int i = 0; i < digits.Count(); i++ )
                {
                    Console.WriteLine( "For " + digits[i].ToString() + " eout = " + eout[i].ToString( "f04" )
                        + " ein = " + ein[i].ToString( "f04" ) );
                }
                Console.ReadLine();
            }
        }


        private static Data[] RawToData( List<double[]> raw_data, List<double> y )
        {
            Data[] training_points = new Data[raw_data.Count()];
            for( int j = 0; j < raw_data.Count(); j++ )
            {
                training_points[j] = new Data( raw_data[j] );
                training_points[j].y = Math.Sign( y[j] );
            }
            return training_points;
        }

        static void DataParseWithX0( string filename, int d, List<double> y, List<double[]> raw_data, int plusone )
        {
            string line;
            using( StreamReader file = new StreamReader( filename ) )
            {
                // if this was real programming, we'd put some error handling around this
                while( ( line = file.ReadLine() ) != null )
                {
                    double[] point = new double[d+1];
                    string[] parts = line.Split( new char[0], StringSplitOptions.RemoveEmptyEntries );
                    double yn = double.Parse( parts[0] );
                    point[0] = 1;
                    if( yn == plusone )
                    {
                        y.Add( 1 );
                        point[1] = double.Parse( parts[1] );
                        point[2] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                    else
                    {
                        y.Add( -1 );
                        point[1] = double.Parse( parts[1] );
                        point[2] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                }
                file.Close();
            }
        }
    }
}

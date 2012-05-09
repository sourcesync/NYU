//
//  ViewController.m
//  CamController
//
//  Created by George Williams on 3/6/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#import "ViewController.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>


@implementation ViewController

//@synthesize ports=_ports;
@synthesize picker=_picker;

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

- (void)viewDidLoad
{
    [super viewDidLoad];
	// Do any additional setup after loading the view, typically from a nib.
    
    self.picker.delegate = self;
    self.picker.dataSource = self;
    
}

- (void)viewDidUnload
{
    [super viewDidUnload];
    // Release any retained subviews of the main view.
    // e.g. self.myOutlet = nil;
}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated
{
	[super viewWillDisappear:animated];
}

- (void)viewDidDisappear:(BOOL)animated
{
	[super viewDidDisappear:animated];
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    // Return YES for supported orientations
    return (interfaceOrientation != UIInterfaceOrientationPortraitUpsideDown);
}

#pragma mark - picker view...

//PickerViewController.m
- (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)thePickerView {
    
    return 2;
}

//PickerViewController.m
- (NSInteger)pickerView:(UIPickerView *)thePickerView numberOfRowsInComponent:(NSInteger)component {
    
    return 10;
}

//PickerViewController.m
- (NSString *)pickerView:(UIPickerView *)thePickerView titleForRow:(NSInteger)row forComponent:(NSInteger)component {
    NSNumber *_num = [ NSNumber numberWithInt:9000 + row ];
    NSString *str = [ NSString stringWithFormat:@"%@", _num ];
    return str;
}

#pragma mark -commands

-(IBAction) startit:(id)sender
{
    int min = 9000 + [ self.picker selectedRowInComponent:0 ];
    int max = 9000 + [ self.picker selectedRowInComponent:1 ];
    
    if ( max<min) return;
    
    for ( int i=min;i<=max;i++)
    {
    
        [ self send:@"2:c:\\tmp\\iphonea" ipAddress:@"10.0.0.3" port:9145 ];
        [ self send:@"3:3" ipAddress:@"10.0.0.3" port:9145 ];
        [ self send:@"0:" ipAddress:@"10.0.0.3" port:9145 ];
    
        [ self send:@"2:c:\\tmp\\iphoneb" ipAddress:@"10.0.0.3" port:9146 ];
        [ self send:@"3:3" ipAddress:@"10.0.0.3" port:9146 ];
        [ self send:@"0:" ipAddress:@"10.0.0.3" port:9146 ];
    }
}


-(IBAction) stopit:(id)sender
{
    [ self send:@"1:" ipAddress:@"10.0.0.3" port:9145 ];
    [ self send:@"1:" ipAddress:@"10.0.0.3" port:9146 ];
}


- (void) send:(NSString *) msg ipAddress:(NSString *) ip port:(int)p
{
    int sock;
    int err = 0;
    
    struct sockaddr_in destination;
    
    sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    NSLog(@"sock = %d\n", sock);
    
    memset(&destination,0,sizeof(destination));
    
    destination.sin_family = AF_INET;
    
    destination.sin_addr.s_addr = inet_addr([ ip UTF8String ] );
    
    destination.sin_port = htons(p);
    
    //err = setsockopt(sock, IPPROTO_IP, IP_MULTICAST_IF, &destination, sizeof(destination) );
    NSLog(@"sets = %d\n", err);
    
    char *cmsg = (char *)[ msg UTF8String];
    unsigned int echolen = strlen(cmsg);
    
    err = sendto(sock, cmsg, echolen, 0, (struct sockaddr *) &destination,
                 sizeof(destination) );
    NSLog(@"send %d\n", err);
    
    
}

@end

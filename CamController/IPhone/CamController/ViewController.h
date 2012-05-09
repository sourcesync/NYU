//
//  ViewController.h
//  CamController
//
//  Created by George Williams on 3/6/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController
    < UIPickerViewDelegate, UIPickerViewDataSource >


//@property (nonatomic, retain) NSMutableArray *ports;
@property (nonatomic, retain) IBOutlet UIPickerView *picker;

- (void) send:(NSString *) msg ipAddress:(NSString *) ip port:(int)p;

-(IBAction) startit:(id)sender;
-(IBAction) stopit:(id)sender;

@end
